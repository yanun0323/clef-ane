#import <Foundation/Foundation.h>
#import <dlfcn.h>
#import <objc/message.h>
int main(void) {
  setvbuf(stdout, NULL, _IONBF, 0);
  @autoreleasepool {
    void *h = dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW);
    printf("dlopen AppleNeuralEngine: %s\n", h ? "ok" : dlerror());
    const char *names[] = {"_ANEClient","_ANECompiler","_ANEInMemoryModelDescriptor","_ANEInMemoryModel","_ANERequest","_ANEIOSurfaceObject","_ANEDeviceInfo",NULL};
    for (int i = 0; names[i]; i++) printf("%-30s %s\n", names[i], NSClassFromString([NSString stringWithUTF8String:names[i]]) ? "present" : "missing");
    void *c = dlopen("/System/Library/Frameworks/CoreAI.framework/CoreAI", RTLD_NOW);
    printf("dlopen CoreAI: %s\n", c ? "ok" : dlerror());
  }
}
