
#ifndef PyImgC_CIMG_NSBITMAPIMAGEREP_PLUGIN_H
#define PyImgC_CIMG_NSBITMAPIMAGEREP_PLUGIN_H

#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <map>

#define SUFFIXED(filename, suffix) !cimg::strncasecmp(cimg::split_filename(filename), suffix, strlen(suffix))

inline bool bitmap_can_load(const char *filename) const {
    @autoreleasepool {
        NSString *suffix = [NSString stringWithUTF8String:cimg::split_filename(filename)];
        UTI *uti = [UTI UTIWithFileExtension:suffix];
        return static_cast<bool>(
            [[UTI imageContentTypes] containsObject:uti]);
    }
    return false;
}
inline bool bitmap_can_load(NSString *filename) const {
    @autoreleasepool {
        NSString *suffix = [filename pathExtension];
        UTI *uti = [UTI UTIWithFileExtension:suffix];
        return static_cast<bool>(
            [[UTI imageContentTypes] containsObject:uti]);
    }
    return false;
}

#define IMGC_NSBITMAP_NOSAVE static_cast<NSBitmapImageFileType>(-666)

inline NSBitmapImageFileType bitmap_can_save(const char *filename) const {
    //return IMGC_NSBITMAP_NOSAVE;
    if (SUFFIXED(filename, "jpg") || SUFFIXED(filename, "jpeg")) { return NSJPEGFileType; }
    if (SUFFIXED(filename, "png"))                               { return NSPNGFileType; }
    if (SUFFIXED(filename, "tif") || SUFFIXED(filename, "tiff")) { return NSTIFFFileType; }
    if (SUFFIXED(filename, "bmp"))                               { return NSBMPFileType; }
    if (SUFFIXED(filename, "jpe") || SUFFIXED(filename, "jpgg")) { return NSJPEGFileType; }
    if (SUFFIXED(filename, "jp2") || SUFFIXED(filename, "jpg2")) { return NSJPEG2000FileType; }
    if (SUFFIXED(filename, "jpe2"))                              { return NSJPEG2000FileType; }
    return IMGC_NSBITMAP_NOSAVE;
}
inline NSBitmapImageFileType bitmap_can_save(NSString *filenameMixedCase) const {
    //return IMGC_NSBITMAP_NOSAVE;
    @autoreleasepool {
        NSString *filename = [filenameMixedCase lowercaseString];
        if ([filename hasSuffix:@"jpg"] || [filename hasSuffix:@"jpeg"]) { return NSJPEGFileType; }
        if ([filename hasSuffix:@"png"])                                 { return NSPNGFileType; }
        if ([filename hasSuffix:@"tif"] || [filename hasSuffix:@"tiff"]) { return NSTIFFFileType; }
        if ([filename hasSuffix:@"bmp"])                                 { return NSBMPFileType; }
        if ([filename hasSuffix:@"jpe"] || [filename hasSuffix:@"jpgg"]) { return NSJPEGFileType; }
        if ([filename hasSuffix:@"jp2"] || [filename hasSuffix:@"jpg2"]) { return NSJPEG2000FileType; }
        if ([filename hasSuffix:@"jpe2"])                                { return NSJPEG2000FileType; }
    }
    return IMGC_NSBITMAP_NOSAVE;
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
    
    const T *const dataPtrI = const_cast<const T *const>(
        static_cast<T *>([bitmap bitmapData]));
    
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
NSBitmapImageRep *get_bitmap(const unsigned z=0) const {
    if (is_empty()) {
        throw CImgArgumentException(_cimg_instance
                                    "get_bitmap() : Empty CImg instance");
    }
    
    if (z >= _depth) {
        throw CImgInstanceException(_cimg_instance
                                    "get_bitmap() : Instance has z-depth >= %u",
                                    z);
    }
    if (_spectrum > 4) {
        cimg::warn(_cimg_instance
                   "get_bitmap() : NSImages don't really support >4 channels -- higher-order dimensions will be ignored");
    }
    
    //NSInteger bps = 8 * sizeof(T);
    NSInteger format = 0;
    format |= isFloat() ? NSFloatingPointSamplesBitmapFormat : 0;
    
    // NSLog(@"Width: %i", _width);
    // NSLog(@"Height: %i", _height);
    // NSLog(@"Spectrum: %i", _spectrum);
    // NSLog(@"Format: %li", (long)format);
    // NSLog(@"BPS: %li", (long)bps);
    
    NSBitmapImageRep *bitmap = [[NSBitmapImageRep alloc]
        initWithBitmapDataPlanes:(unsigned char **)&_data
        pixelsWide:nsWidth()
        pixelsHigh:nsHeight()
        bitsPerSample:itemsize()
        samplesPerPixel:nsSpectrum()
        hasAlpha:NO
        isPlanar:YES
        colorSpaceName:NSDeviceRGBColorSpace
        bitmapFormat:format
        bytesPerRow:0 //(NSInteger)(_width * _spectrum * sizeof(T))
        bitsPerPixel:0]; //(NSInteger)_spectrum * bps
    
    
    // NSBitmapImageRep *bitmap = [[NSBitmapImageRep alloc]
    //     initWithBitmapDataPlanes:(unsigned char **)&_data   /// for once no switching on typecode!
    //     pixelsWide:(NSInteger)_width
    //     pixelsHigh:(NSInteger)_height
    //     bitsPerSample:(NSInteger)sizeof(T)
    //     samplesPerPixel:(NSInteger)_spectrum
    //     hasAlpha:NO
    //     isPlanar:NO
    //     colorSpaceName:NSDeviceRGBColorSpace    /// for now...
    //     bitmapFormat:format
    //     bytesPerRow:0                           /// = "you figure it out"
    //     bitsPerPixel:0];                        /// == "bitsPerPixel > my pay grade"
    
    return bitmap;
}

static CImg<T> get_load_quartz(const char *filename) {
    NSString *nsfilename = [NSString stringWithUTF8String:filename];
    const NSBitmapImageRep *bitmap = [
        NSBitmapImageRep imageRepWithContentsOfFile:nsfilename];
    [nsfilename autorelease];
    return CImg<T>(bitmap);
}
static CImg<T> get_load_quartz(NSString *filename) {
    const NSBitmapImageRep *bitmap = [
        NSBitmapImageRep imageRepWithContentsOfFile:filename];
    return CImg<T>(bitmap);
}

CImg& load_quartz(const char *filename) {
    return get_load_quartz(filename).swap(*this);
}
CImg& load_quartz(NSString *filename) {
    return get_load_quartz(filename).swap(*this);
}

const CImg& save_quartz(const char *filename, NSBitmapImageFileType filetype=NSJPEGFileType) const {
    /// THIS DOES NOT WORK
    /// (unless by 'work' you mean 'BUS ERROR 10')
    NSBitmapImageRep *bitmap = (NSBitmapImageRep *)get_bitmap();
    NSData *out = [bitmap representationUsingType:filetype properties:nil];
    [out writeToFile:[NSString stringWithUTF8String:filename] atomically:NO];
    [bitmap autorelease];
    [out autorelease];
    return *this;
}

const CImg& save_coregraphics(CFURLRef url) const {
    /// kUTTypeJPEG
    CGImageRef imgref = cgImageRef(cgContext(CGColorSpaceCreateDeviceRGB()));
    CGImageDestinationRef destref = CGImageDestinationCreateWithURL(url,
        CFSTR("public.jpeg"), 1, NULL);
    if (destref) {
        CGImageDestinationAddImage(destref, imgref, NULL);
        CGImageDestinationFinalize(destref);
        CFRelease(destref);
    }
    CGImageRelease(imgref);
    return *this;
}
const CImg& save_coregraphics(CFStringRef filename) const {
    CFURLRef url = CFURLCreateWithFileSystemPath(NULL, filename, kCFURLPOSIXPathStyle, 0);
    save_coregraphics(url);
    CFRelease(url);
    return *this;
}
const CImg& save_coregraphics(const char *filename) const {
    CFStringRef pth = CFStringCreateWithCString(NULL, filename, kCFStringEncodingUTF8);
    save_coregraphics(pth);
    CFRelease(pth);
    return *this;
}

#ifndef cimg_load_plugin4
#define cimg_load_plugin4(filename) \
    if (bitmap_can_load(filename)) { return load_quartz(filename); }
#endif

// #ifndef cimg_save_plugin4
// #define cimg_save_plugin4(filename) \
//     NSBitmapImageFileType filetype = bitmap_can_save(filename); \
//     if (filetype != IMGC_NSBITMAP_NOSAVE) { return save_coregraphics(filename); }
// #endif

// #ifndef cimg_save_plugin4
// #define cimg_save_plugin4(filename) \
//     NSBitmapImageFileType filetype = bitmap_can_save(filename); \
//     if (filetype != IMGC_NSBITMAP_NOSAVE) { return save_quartz(filename, filetype); }
// #endif

#endif /// PyImgC_CIMG_NSBITMAPIMAGEREP_PLUGIN_H
