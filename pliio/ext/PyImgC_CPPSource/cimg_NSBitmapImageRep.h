
#ifdef __OBJC__
#import <Cocoa/Cocoa.h>
#ifndef PyImgC_CIMG_NSBITMAPIMAGEREP_PLUGIN_H
#define PyImgC_CIMG_NSBITMAPIMAGEREP_PLUGIN_H

//----------------------------
// NSBitmapImageRep-to-CImg conversion
//----------------------------
/// Copy constructor
CImg(const NSBitmapImageRep *const bitmap):_width(0),_height(0),_depth(0),_spectrum(0),_is_shared(false),_data(0) {
    assign(bitmap);
}
CImg(NSBitmapImageRep *bitmap):_width(0),_height(0),_depth(0),_spectrum(0),_is_shared(false),_data(0) {
    assign(const_cast<NSBitmapImageRep *>(bitmap));
}

// In-place constructor
CImg<T> &assign(NSBitmapImageRep *bitmap) const {
    if (!bitmap) return assign();
    
    /// N.B. should we be, like, aborting the mission if T*
    /// happens to be something other than some form of char??
    const unsigned char *const dataPtrI = const_cast<const unsigned char *const>(
        static_cast<unsigned char *>([bitmap bitmapData]));
    
    int nChannels = [bitmap samplesPerPixel],
        W = [bitmap pixelsWide],
        H = [bitmap pixelsHigh];
    
    assign(dataPtrI,
        const_cast<int&>(W),
        const_cast<int&>(H), 1,
        const_cast<int&>(nChannels), true);
    
    return *this;
}

//----------------------------
// CImg-to-NSBitmapImageRep conversion
//----------------------------
// z is the z-coordinate of the CImg slice that one wants to copy.
NSBitmapImageRep *getBitmapImageRep(const unsigned z=0) {
    if (is_empty()) {
        throw CImgArgumentException(_cimg_instance
                                    "getBitmapImageRep() : Empty CImg instance.",
                                    cimg_instance);
    }
    
    if (z >= _depth) {
        throw CImgInstanceException(_cimg_instance
                                    "getBitmapImageRep() : Instance has not Z-dimension %u.",
                                    cimg_instance,
                                    z);
    }
    if (_spectrum > 4) {
        cimg::warn(_cimg_instance
                   "getBitmapImageRep() : NSImage don't really support >4 channels -- higher-order dimensions will be ignored.",
                   cimg_instance);
    }
    
    NSBitmapImageRep *bitmap = [[NSBitmapImageRep alloc]
        initWithBitmapDataPlanes:(unsigned char **)&_data   /// for once no switching on typecode!
        pixelsWide:_width
        pixelsHigh:_height
        bitsPerSample:sizeof(T)
        samplesPerPixel:_spectrum
        hasAlpha:NO
        isPlanar:NO                             /// typedef NO HELL_NO
        colorSpaceName:NSDeviceRGBColorSpace    /// for now...
        bitmapFormat:0                          /// DO NOT FUCK WITH THIS ZERO
        bytesPerRow:0                           /// = "you figure it out"
        bitsPerPixel:0];                        /// == "bitsPerPixel > my pay grade"
    
    return bitmap;
    
}

#endif /// PyImgC_CIMG_NSBITMAPIMAGEREP_PLUGIN_H
#endif /// __OBJC__