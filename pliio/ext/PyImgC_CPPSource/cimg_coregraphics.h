
#ifdef __OBJC__
#import <Cocoa/Cocoa.h>
#import <Foundation/Foundation.h>
#import <CoreFoundation/CoreFoundation.h>
#ifndef PyImgC_CIMG_NSBITMAPIMAGEREP_PLUGIN_H
#define PyImgC_CIMG_NSBITMAPIMAGEREP_PLUGIN_H

#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <map>

#define FN(alias, func, returns) \
template <typename ...Args> \
struct alias { \
    returns operator()(Args&&... args) { \
        return func(args...); \
    } \
};


template <typename RefType,
          typename RefRelease=CFRelease>
struct Ref {
    Ref() : r(nullptr), inUse(false) {}
    explicit Ref(RefType ref) : r(ref), inUse(true) {}
    Ref(const Ref &rhs) : inUse(true) { r = rhs.r; }
    Ref &operator=(const Ref &rhs) : inUse(true) {
        if (this != rhs) {
            if (r) { RefRelease(r); }
            r = rhs.r;
        }
        return *this;
    }
    
    Ref(Ref &&rhs) = delete;
    Ref &operator=(Ref &&rhs) = delete;
    
    ~Ref() { if (inUse) { deref(); } }
    void deref() {
        RefRelease(r);
        inUse = false;
    }
    
    RefType get() const { return r; }
    void set(RefType rhs) {
        if (r != rhs) {
            if (r) { RefRelease(r); }
            r = rhs;
        }
    }
    void Release() {
        RefRelease(r);
        r = nullptr;
        inUse = false;
    }
    
private:
    bool inUse;
    RefType r;
};

namespace cf {
    namespace array {
        typedef Ref<CFArrayRef> ref;
    }
    namespace string {
        typedef Ref<CFStringRef> ref;
    }
    namespace URL {
        typedef Ref<CFURLRef> ref;
    }
}

namespace cg {
    namespace bitmapContext {
        typedef CGContextRef refType;
        typedef Ref<refType, CGContextRelease> ref;
        FN(Create, CGBitmapContextCreate, refType);
    }
    namespace colorspace {
        typedef CGColorSpaceRef refType;
        typedef Ref<refType, CGColorSpaceRelease> ref;
        FN(ComponentCount, CGColorSpaceGetNumberOfComponents, refType);
        
    }
}







#endif /// PyImgC_CIMG_NSBITMAPIMAGEREP_PLUGIN_H
#endif /// __OBJC__