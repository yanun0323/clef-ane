// Minimal C bridge over the private _ANEInMemoryModel API, driven from Python through ctypes.
// Everything compiled through here runs on the ANE or fails to compile; there is no GPU/CPU fallback.
// Built without ARC: Kernel keeps Objective-C objects in a C struct, so retain/release are explicit.
#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#import <dlfcn.h>
#import <objc/message.h>

#define QOS 21
#define MSG(ret, ...) ((ret(*)(__VA_ARGS__))objc_msgSend)

static Class gDesc, gModel, gRequest, gIO;

typedef struct {
  id model, request;
  NSString *dir;
  int nIn, nOut;
  IOSurfaceRef in[16], out[16];
  size_t inBytes[16], outBytes[16];
} Kernel;

static void setError(char *err, size_t len, NSString *text) {
  if (err && len) snprintf(err, len, "%s", text.UTF8String);
}

int ane_init(void) {
  if (gDesc) return 0;
  if (!dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW)) return -1;
  gDesc = NSClassFromString(@"_ANEInMemoryModelDescriptor");
  gModel = NSClassFromString(@"_ANEInMemoryModel");
  gRequest = NSClassFromString(@"_ANERequest");
  gIO = NSClassFromString(@"_ANEIOSurfaceObject");
  return (gDesc && gModel && gRequest && gIO) ? 0 : -2;
}

static IOSurfaceRef makeSurface(size_t bytes) {
  return IOSurfaceCreate((__bridge CFDictionaryRef)@{
    (id)kIOSurfaceWidth: @(bytes), (id)kIOSurfaceHeight: @1, (id)kIOSurfaceBytesPerElement: @1,
    (id)kIOSurfaceBytesPerRow: @(bytes), (id)kIOSurfaceAllocSize: @(bytes), (id)kIOSurfacePixelFormat: @0});
}

void ane_free(void *handle);

// Weights are passed as parallel arrays; names look like "@model_path/weights/foo.bin".
void *ane_compile(const char *mil, size_t milLen, int nWeights, const char **names, const uint8_t **datas,
                  const size_t *lens, int nIn, const size_t *inBytes, int nOut, const size_t *outBytes,
                  char *err, size_t errLen) {
  @autoreleasepool {
    if (nIn < 1 || nIn > 16 || nOut < 1 || nOut > 16 || nWeights < 0 || nWeights > 64) {
      setError(err, errLen, @"invalid ANE buffer count"); return NULL;
    }
    const size_t maxBufferBytes = 256 * 1024 * 1024;
    for (int i = 0; i < nIn; i++) if (!inBytes[i] || inBytes[i] > maxBufferBytes) {
      setError(err, errLen, @"invalid ANE input buffer size"); return NULL;
    }
    for (int i = 0; i < nOut; i++) if (!outBytes[i] || outBytes[i] > maxBufferBytes) {
      setError(err, errLen, @"invalid ANE output buffer size"); return NULL;
    }
    if (ane_init() != 0) { setError(err, errLen, @"private ANE classes unavailable"); return NULL; }
    NSData *milData = [NSData dataWithBytes:mil length:milLen];
    NSMutableDictionary *weights = [NSMutableDictionary dictionary];
    for (int i = 0; i < nWeights; i++)
      weights[@(names[i])] = @{@"offset": @0, @"data": [NSData dataWithBytes:datas[i] length:lens[i]]};

    id desc = MSG(id, id, SEL, id, id, id)(gDesc, NSSelectorFromString(@"modelWithMILText:weights:optionsPlist:"),
                                           milData, weights, nil);
    id model = desc ? MSG(id, id, SEL, id)(gModel, NSSelectorFromString(@"inMemoryModelWithDescriptor:"), desc) : nil;
    if (!model) { setError(err, errLen, @"could not create in-memory model (bad MIL text?)"); return NULL; }

    // The compiler reads the MIL and weights back from a directory named after the model's hash.
    NSString *hex = MSG(id, id, SEL)(model, NSSelectorFromString(@"hexStringIdentifier"));
    NSString *dir = [NSTemporaryDirectory() stringByAppendingPathComponent:hex];
    NSFileManager *fm = NSFileManager.defaultManager;
    [fm createDirectoryAtPath:[dir stringByAppendingPathComponent:@"weights"] withIntermediateDirectories:YES attributes:nil error:nil];
    [milData writeToFile:[dir stringByAppendingPathComponent:@"model.mil"] atomically:YES];
    for (int i = 0; i < nWeights; i++) {
      NSString *rel = @(names[i]);
      if ([rel hasPrefix:@"@model_path/"]) rel = [rel substringFromIndex:12];
      NSString *path = [dir stringByAppendingPathComponent:rel];
      [fm createDirectoryAtPath:path.stringByDeletingLastPathComponent withIntermediateDirectories:YES attributes:nil error:nil];
      [[NSData dataWithBytes:datas[i] length:lens[i]] writeToFile:path atomically:YES];
    }

    NSError *e = nil;
    BOOL ok = MSG(BOOL, id, SEL, unsigned, id, NSError **)(model, NSSelectorFromString(@"compileWithQoS:options:error:"), QOS, @{}, &e);
    if (!ok) { setError(err, errLen, [@"compile: " stringByAppendingString:e.description ?: @"unknown"]); [fm removeItemAtPath:dir error:nil]; return NULL; }
    ok = MSG(BOOL, id, SEL, unsigned, id, NSError **)(model, NSSelectorFromString(@"loadWithQoS:options:error:"), QOS, @{}, &e);
    if (!ok) { setError(err, errLen, [@"load: " stringByAppendingString:e.description ?: @"unknown"]); [fm removeItemAtPath:dir error:nil]; return NULL; }

    Kernel *k = calloc(1, sizeof(Kernel));
    k->model = [model retain]; k->dir = [dir retain]; k->nIn = nIn; k->nOut = nOut;
    NSMutableArray *ins = [NSMutableArray array], *inIdx = [NSMutableArray array];
    NSMutableArray *outs = [NSMutableArray array], *outIdx = [NSMutableArray array];
    SEL wrap = NSSelectorFromString(@"objectWithIOSurface:");
    for (int i = 0; i < nIn; i++) {
      k->inBytes[i] = inBytes[i]; k->in[i] = makeSurface(inBytes[i]);
      id surface = k->in[i] ? MSG(id, id, SEL, IOSurfaceRef)(gIO, wrap, k->in[i]) : nil;
      if (!surface) { setError(err, errLen, @"could not allocate ANE input surface"); ane_free(k); return NULL; }
      [ins addObject:surface]; [inIdx addObject:@(i)];
    }
    for (int i = 0; i < nOut; i++) {
      k->outBytes[i] = outBytes[i]; k->out[i] = makeSurface(outBytes[i]);
      id surface = k->out[i] ? MSG(id, id, SEL, IOSurfaceRef)(gIO, wrap, k->out[i]) : nil;
      if (!surface) { setError(err, errLen, @"could not allocate ANE output surface"); ane_free(k); return NULL; }
      [outs addObject:surface]; [outIdx addObject:@(i)];
    }
    k->request = MSG(id, id, SEL, id, id, id, id, id, id, id)(gRequest,
      NSSelectorFromString(@"requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:"),
      ins, inIdx, outs, outIdx, nil, nil, @0);
    [k->request retain];
    return k;
  }
}

int ane_write_input(void *handle, int idx, const void *data, size_t bytes) {
  Kernel *k = handle;
  if (idx < 0 || idx >= k->nIn || bytes != k->inBytes[idx]) return -1;
  IOSurfaceLock(k->in[idx], 0, NULL);
  memcpy(IOSurfaceGetBaseAddress(k->in[idx]), data, bytes);
  IOSurfaceUnlock(k->in[idx], 0, NULL);
  return 0;
}

// Overwrite part of an input buffer in place, leaving the rest as it was (used for the persistent KV cache).
int ane_write_input_at(void *handle, int idx, size_t offset, const void *data, size_t bytes) {
  Kernel *k = handle;
  if (idx < 0 || idx >= k->nIn || offset > k->inBytes[idx] || bytes > k->inBytes[idx] - offset) return -1;
  IOSurfaceLock(k->in[idx], 0, NULL);
  memcpy((uint8_t *)IOSurfaceGetBaseAddress(k->in[idx]) + offset, data, bytes);
  IOSurfaceUnlock(k->in[idx], 0, NULL);
  return 0;
}

int ane_read_output(void *handle, int idx, void *dst, size_t bytes) {
  Kernel *k = handle;
  if (idx < 0 || idx >= k->nOut || bytes != k->outBytes[idx]) return -1;
  IOSurfaceLock(k->out[idx], kIOSurfaceLockReadOnly, NULL);
  memcpy(dst, IOSurfaceGetBaseAddress(k->out[idx]), bytes);
  IOSurfaceUnlock(k->out[idx], kIOSurfaceLockReadOnly, NULL);
  return 0;
}

int ane_eval(void *handle, char *err, size_t errLen) {
  Kernel *k = handle;
  @autoreleasepool {
    NSError *e = nil;
    BOOL ok = MSG(BOOL, id, SEL, unsigned, id, id, NSError **)(k->model,
      NSSelectorFromString(@"evaluateWithQoS:options:request:error:"), QOS, @{}, k->request, &e);
    if (!ok) setError(err, errLen, [@"eval: " stringByAppendingString:e.description ?: @"unknown"]);
    return ok ? 0 : -1;
  }
}

void ane_free(void *handle) {
  Kernel *k = handle;
  if (!k) return;
  @autoreleasepool {
    NSError *e = nil;
    MSG(BOOL, id, SEL, unsigned, NSError **)(k->model, NSSelectorFromString(@"unloadWithQoS:error:"), QOS, &e);
    for (int i = 0; i < k->nIn; i++) if (k->in[i]) CFRelease(k->in[i]);
    for (int i = 0; i < k->nOut; i++) if (k->out[i]) CFRelease(k->out[i]);
    [NSFileManager.defaultManager removeItemAtPath:k->dir error:nil];
    [k->model release]; [k->request release]; [k->dir release];
    free(k);
  }
}
