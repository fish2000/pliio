
#ifdef __OBJC__
#import <Cocoa/Cocoa.h>
#ifndef PyImgC_CIMG_NSBITMAPIMAGEREP_PLUGIN_H
#define PyImgC_CIMG_NSBITMAPIMAGEREP_PLUGIN_H

#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <map>

#define SUFFIXED(filename, suffix) !cimg::strncasecmp(cimg::split_filename(filename), suffix, strlen(suffix))

inline bool bitmap_can_load(const char *filename) {
    char *imagetype;
    @autoreleasepool {
        NSEnumerator *imagetypes = [[NSBitmapImageRep imageFileTypes] objectEnumerator];
        while ((imagetype = (char *)[[imagetypes nextObject] UTF8String])) {
            if (SUFFIXED(filename, imagetype)) {
                return true;
            }
        }
    }
    return false;
}

inline NSBitmapImageFileType bitmap_can_save(const char *filename) {
    if (SUFFIXED(filename, "jpg") || SUFFIXED(filename, "jpeg")) { return NSJPEGFileType; }
    if (SUFFIXED(filename, "png"))                               { return NSPNGFileType; }
    if (SUFFIXED(filename, "tif") || SUFFIXED(filename, "tiff")) { return NSTIFFFileType; }
    if (SUFFIXED(filename, "jpe") || SUFFIXED(filename, "jpgg")) { return NSJPEGFileType; }
    if (SUFFIXED(filename, "bmp")) { return NSBMPFileType; }
    if (SUFFIXED(filename, "jpe") || SUFFIXED(filename, "jpe2")) { return NSJPEG2000FileType; }
    if (SUFFIXED(filename, "jp2") || SUFFIXED(filename, "jpg2")) { return NSJPEG2000FileType; }
    return static_cast<NSBitmapImageFileType>(-1);
}

//---------------------------------------
// NSBitmapImageRep-to-CImg conversion
//---------------------------------------
/// Copy constructor
CImg<T>(const NSBitmapImageRep *bitmap):_width(0),_height(0),_depth(0),_spectrum(0),_is_shared(false),_data(0) {
    assign(bitmap);
}
CImg(NSBitmapImageRep *bitmap):_width(0),_height(0),_depth(0),_spectrum(0),_is_shared(false),_data(0) {
    assign(const_cast<NSBitmapImageRep *>(bitmap));
}

// In-place constructor
CImg<T> &assign(const NSBitmapImageRep *bitmap) {
    if (!bitmap) return *this;
    
    /// N.B. should we be, like, aborting the mission if T*
    /// happens to be something other than some form of char??
    const unsigned char *const dataPtrI = const_cast<const unsigned char *const>(
        static_cast<unsigned char *>([bitmap bitmapData]));
    
    int nChannels = (int)[bitmap samplesPerPixel],
        W = (int)[bitmap pixelsWide],
        H = (int)[bitmap pixelsHigh];
    
    assign(dataPtrI,
        const_cast<int&>(W),
        const_cast<int&>(H), 1,
        const_cast<int&>(nChannels), true);
    
    return *this;
}

//---------------------------------------
// CImg-to-NSBitmapImageRep conversion
//---------------------------------------
NSBitmapImageRep *get_bitmap(const unsigned z=0) {
    if (is_empty()) {
        throw CImgArgumentException(_cimg_instance
                                    "get_bitmap() : Empty CImg instance.");
    }
    
    if (z >= _depth) {
        throw CImgInstanceException(_cimg_instance
                                    "get_bitmap() : Instance has not Z-dimension %u.",
                                    z);
    }
    if (_spectrum > 4) {
        cimg::warn(_cimg_instance
                   "get_bitmap() : NSImage don't really support >4 channels -- higher-order dimensions will be ignored.");
    }
    
    NSInteger format = 0;
    format |= std::is_floating_point<T>::value ? NSFloatingPointSamplesBitmapFormat : 0;
    
    NSBitmapImageRep *bitmap = [[NSBitmapImageRep alloc]
        initWithBitmapDataPlanes:(unsigned char **)&_data   /// for once no switching on typecode!
        pixelsWide:(NSInteger)_width
        pixelsHigh:(NSInteger)_height
        bitsPerSample:(NSInteger)sizeof(T)
        samplesPerPixel:(NSInteger)_spectrum
        hasAlpha:NO
        isPlanar:NO
        colorSpaceName:NSDeviceRGBColorSpace    /// for now...
        bitmapFormat:format
        bytesPerRow:0                           /// = "you figure it out"
        bitsPerPixel:0];                        /// == "bitsPerPixel > my pay grade"
    
    return bitmap;
}

static CImg<T> get_load_quartz(const char *filename) {
    const NSBitmapImageRep *bitmap = [
        NSBitmapImageRep imageRepWithContentsOfFile:[
            [NSString alloc] initWithUTF8String:filename]];
    return CImg<T>(bitmap);
}

CImg& load_quartz(const char *filename) {
    return get_load_quartz(filename).swap(*this);
}

const CImg& save_quartz(const char *filename,
                        NSBitmapImageFileType filetype=NSJPEGFileType) const {
    @autoreleasepool {
        const NSBitmapImageRep *bitmap = get_bitmap();
        const NSData *out = [bitmap representationUsingType:filetype properties:nil];
        [out writeToFile:[[NSString alloc] initWithUTF8String:filename] atomically:NO];
    }
    return *this;
}

#ifndef cimg_load_plugin4
#define cimg_load_plugin4(filename) \
    if (bitmap_can_load(filename)) { return load_quartz(filename); }
#endif
    
#ifndef cimg_save_plugin4
#define cimg_save_plugin4(filename) \
    NSBitmapImageFileType bitmap_format = bitmap_can_save(filename); \
    if (bitmap_format > 0) { return save_quartz(filename, bitmap_format); }
#endif

#endif /// PyImgC_CIMG_NSBITMAPIMAGEREP_PLUGIN_H
#endif /// __OBJC__