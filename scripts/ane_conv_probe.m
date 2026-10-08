// Feasibility probe: compile and run one 1x1 conv (= linear layer) on the ANE through the
// private _ANEInMemoryModel path, check the numbers, and time it.
// Usage: ane_conv_probe [in_ch out_ch spatial iters]
#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#import <dlfcn.h>
#import <mach/mach_time.h>
#import <objc/message.h>

static NSString *milConv(int in, int out, int sp) {
  return [NSString stringWithFormat:
    @"program(1.3)\n"
    "[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, "
    "{\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, "
    "{\"coremltools-version\", \"9.0\"}})]\n"
    "{\n"
    "    func main<ios18>(tensor<fp32, [1, %d, 1, %d]> x) {\n"
    "        string c_pad_type = const()[name = string(\"c_pad_type\"), val = string(\"valid\")];\n"
    "        tensor<int32, [2]> c_strides = const()[name = string(\"c_strides\"), val = tensor<int32, [2]>([1, 1])];\n"
    "        tensor<int32, [4]> c_pad = const()[name = string(\"c_pad\"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n"
    "        tensor<int32, [2]> c_dilations = const()[name = string(\"c_dilations\"), val = tensor<int32, [2]>([1, 1])];\n"
    "        int32 c_groups = const()[name = string(\"c_groups\"), val = int32(1)];\n"
    "        string to_fp16 = const()[name = string(\"to_fp16\"), val = string(\"fp16\")];\n"
    "        tensor<fp16, [1, %d, 1, %d]> x16 = cast(dtype = to_fp16, x = x)[name = string(\"cast_in\")];\n"
    "        tensor<fp16, [%d, %d, 1, 1]> W = const()[name = string(\"W\"), "
    "val = tensor<fp16, [%d, %d, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), offset = uint64(64)))];\n"
    "        tensor<fp16, [1, %d, 1, %d]> y16 = conv(dilations = c_dilations, groups = c_groups, "
    "pad = c_pad, pad_type = c_pad_type, strides = c_strides, weight = W, x = x16)[name = string(\"conv\")];\n"
    "        string to_fp32 = const()[name = string(\"to_fp32\"), val = string(\"fp32\")];\n"
    "        tensor<fp32, [1, %d, 1, %d]> y = cast(dtype = to_fp32, x = y16)[name = string(\"cast_out\")];\n"
    "    } -> (y);\n"
    "}\n",
    in, sp, in, sp, out, in, out, in, out, sp, out, sp];
}

// 64-byte file header + 64-byte chunk header + fp16 data. The BLOBFILE offset (64) points at the chunk header.
static NSData *weightBlob(const _Float16 *w, size_t count) {
  size_t bytes = count * 2, total = 128 + bytes;
  uint8_t *buf = calloc(total, 1);
  buf[0] = 0x01; buf[4] = 0x02;
  uint8_t *chunk = buf + 64;
  chunk[0] = 0xEF; chunk[1] = 0xBE; chunk[2] = 0xAD; chunk[3] = 0xDE; chunk[4] = 0x01;
  *(uint32_t *)(chunk + 8) = (uint32_t)bytes;
  *(uint32_t *)(chunk + 16) = 128;
  memcpy(buf + 128, w, bytes);
  return [NSData dataWithBytesNoCopy:buf length:total freeWhenDone:YES];
}

static IOSurfaceRef surface(size_t bytes) {
  return IOSurfaceCreate((__bridge CFDictionaryRef)@{
    (id)kIOSurfaceWidth: @(bytes), (id)kIOSurfaceHeight: @1, (id)kIOSurfaceBytesPerElement: @1,
    (id)kIOSurfaceBytesPerRow: @(bytes), (id)kIOSurfaceAllocSize: @(bytes), (id)kIOSurfacePixelFormat: @0});
}

static BOOL has(id target, const char *sel) {
  BOOL ok = [target respondsToSelector:NSSelectorFromString(@(sel))];
  printf("  %-80s %s\n", sel, ok ? "ok" : "MISSING");
  return ok;
}

#define MSG(ret, ...) ((ret(*)(__VA_ARGS__))objc_msgSend)

int main(int argc, char **argv) {
  setvbuf(stdout, NULL, _IONBF, 0);
  int in = argc > 1 ? atoi(argv[1]) : 256, out = argc > 2 ? atoi(argv[2]) : 256;
  int sp = argc > 3 ? atoi(argv[3]) : 64, iters = argc > 4 ? atoi(argv[4]) : 50;
  @autoreleasepool {
    mach_timebase_info_data_t tb; mach_timebase_info(&tb);
    if (!dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW)) {
      printf("cannot load AppleNeuralEngine\n"); return 1;
    }
    Class Desc = NSClassFromString(@"_ANEInMemoryModelDescriptor"), IMM = NSClassFromString(@"_ANEInMemoryModel");
    Class Req = NSClassFromString(@"_ANERequest"), IOS = NSClassFromString(@"_ANEIOSurfaceObject");
    printf("selectors:\n");
    BOOL all = has(Desc, "modelWithMILText:weights:optionsPlist:") & has(IMM, "inMemoryModelWithDescriptor:")
      & has(Req, "requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:")
      & has(IOS, "objectWithIOSurface:");
    if (!all) return 2;

    size_t n = (size_t)out * in;
    _Float16 *w = calloc(n, 2);
    int dense = getenv("DENSE") != NULL;  // DENSE=1: random weights, so zero-skipping can't flatter the timing
    if (dense) { srand(7); for (size_t i = 0; i < n; i++) w[i] = (_Float16)(((float)rand() / (float)RAND_MAX - 0.5f) * 0.05f); }
    else for (int o = 0; o < out; o++) w[(size_t)o * in + (o % in)] = (_Float16)0.5f;  // y[o] = 0.5 * x[o % in]
    NSData *mil = [milConv(in, out, sp) dataUsingEncoding:NSUTF8StringEncoding];
    NSData *blob = weightBlob(w, n);

    id desc = MSG(id, id, SEL, id, id, id)(Desc, NSSelectorFromString(@"modelWithMILText:weights:optionsPlist:"), mil,
      @{@"@model_path/weights/weight.bin": @{@"offset": @0, @"data": blob}}, nil);
    id model = MSG(id, id, SEL, id)(IMM, NSSelectorFromString(@"inMemoryModelWithDescriptor:"), desc);
    printf("descriptor: %s, model: %s\n", desc ? "ok" : "nil", model ? "ok" : "nil");
    if (!model) return 3;
    has(model, "compileWithQoS:options:error:"); has(model, "loadWithQoS:options:error:");
    has(model, "evaluateWithQoS:options:request:error:"); has(model, "unloadWithQoS:error:");

    NSString *hex = MSG(id, id, SEL)(model, NSSelectorFromString(@"hexStringIdentifier"));
    NSString *dir = [NSTemporaryDirectory() stringByAppendingPathComponent:hex];
    NSFileManager *fm = NSFileManager.defaultManager;
    [fm createDirectoryAtPath:[dir stringByAppendingPathComponent:@"weights"] withIntermediateDirectories:YES attributes:nil error:nil];
    [mil writeToFile:[dir stringByAppendingPathComponent:@"model.mil"] atomically:YES];
    [blob writeToFile:[dir stringByAppendingPathComponent:@"weights/weight.bin"] atomically:YES];

    NSError *e = nil;
    uint64_t t0 = mach_absolute_time();
    BOOL ok = MSG(BOOL, id, SEL, unsigned, id, NSError **)(model, NSSelectorFromString(@"compileWithQoS:options:error:"), 21, @{}, &e);
    double compileMs = (double)(mach_absolute_time() - t0) * tb.numer / tb.denom / 1e6;
    printf("compile (%dx%d, sp=%d): %s  %.0f ms\n", in, out, sp, ok ? "ok" : "FAILED", compileMs);
    if (!ok) { printf("  %s\n", e.description.UTF8String); [fm removeItemAtPath:dir error:nil]; return 4; }
    ok = MSG(BOOL, id, SEL, unsigned, id, NSError **)(model, NSSelectorFromString(@"loadWithQoS:options:error:"), 21, @{}, &e);
    printf("load: %s\n", ok ? "ok" : "FAILED");
    if (!ok) { printf("  %s\n", e.description.UTF8String); [fm removeItemAtPath:dir error:nil]; return 5; }

    size_t inBytes = (size_t)in * sp * 4, outBytes = (size_t)out * sp * 4;
    IOSurfaceRef sIn = surface(inBytes), sOut = surface(outBytes);
    float *x = malloc(inBytes);
    srand(1);
    for (size_t i = 0; i < (size_t)in * sp; i++) x[i] = (float)rand() / (float)RAND_MAX - 0.5f;
    IOSurfaceLock(sIn, 0, NULL); memcpy(IOSurfaceGetBaseAddress(sIn), x, inBytes); IOSurfaceUnlock(sIn, 0, NULL);
    id wIn = MSG(id, id, SEL, IOSurfaceRef)(IOS, NSSelectorFromString(@"objectWithIOSurface:"), sIn);
    id wOut = MSG(id, id, SEL, IOSurfaceRef)(IOS, NSSelectorFromString(@"objectWithIOSurface:"), sOut);
    id req = MSG(id, id, SEL, id, id, id, id, id, id, id)(Req,
      NSSelectorFromString(@"requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:"),
      @[wIn], @[@0], @[wOut], @[@0], nil, nil, @0);

    SEL ev = NSSelectorFromString(@"evaluateWithQoS:options:request:error:");
    ok = MSG(BOOL, id, SEL, unsigned, id, id, NSError **)(model, ev, 21, @{}, req, &e);
    printf("evaluate: %s\n", ok ? "ok" : "FAILED");
    if (!ok) { printf("  %s\n", e.description.UTF8String); }
    else {
      IOSurfaceLock(sOut, kIOSurfaceLockReadOnly, NULL);
      const float *y = IOSurfaceGetBaseAddress(sOut);
      double maxErr = 0;
      for (int o = 0; o < out; o += (dense ? out / 16 : 1)) for (int s = 0; s < sp; s++) {
        double want = 0;
        if (dense) for (int i = 0; i < in; i++) want += (double)w[(size_t)o * in + i] * x[(size_t)i * sp + s];
        else want = 0.5 * x[(size_t)(o % in) * sp + s];
        double got = y[(size_t)o * sp + s];
        maxErr = fmax(maxErr, fabs(want - got));
      }
      IOSurfaceUnlock(sOut, kIOSurfaceLockReadOnly, NULL);
      printf("max abs error vs CPU: %.5f  (%s)\n", maxErr, maxErr < 1e-2 ? "numbers correct" : "WRONG OUTPUT");
      for (int i = 0; i < 5; i++) MSG(BOOL, id, SEL, unsigned, id, id, NSError **)(model, ev, 21, @{}, req, &e);
      t0 = mach_absolute_time();
      for (int i = 0; i < iters; i++) MSG(BOOL, id, SEL, unsigned, id, id, NSError **)(model, ev, 21, @{}, req, &e);
      double ms = (double)(mach_absolute_time() - t0) * tb.numer / tb.denom / 1e6 / iters;
      printf("%.3f ms/eval  %.2f TFLOPS\n", ms, 2.0 * in * out * sp / 1e9 / ms);
    }
    MSG(BOOL, id, SEL, unsigned, NSError **)(model, NSSelectorFromString(@"unloadWithQoS:error:"), 21, &e);
    CFRelease(sIn); CFRelease(sOut);
    [fm removeItemAtPath:dir error:nil];
  }
  return 0;
}
