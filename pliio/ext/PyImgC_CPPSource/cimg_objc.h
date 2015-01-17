
/*
CImg<T>& load(NSString *filename) { return load([filename UTF8String]); }
static CImg<T> get_load(NSString *filename) { return get_load([filename UTF8String]); }
const CImg<T>& save(NSString *filename,
                    const int number=-1,
                    const unsigned int digits=6) const {
                        return save([filename UTF8String], number, digits);
                    }
*/

inline NSSize nsSize() const { return NSMakeSize(width(), height()); }
inline NSRect nsRect() const { return NSMakeRect({0, 0}, nsSize()); }
inline NSInteger nsWidth() const { return static_cast<NSInteger>(width()); }
inline NSInteger nsHeight() const { return static_cast<NSInteger>(height()); }
inline NSInteger nsSpectrum() const { return static_cast<NSInteger>(spectrum()); }

inline CGSize cgSize() const { return CGSizeMake(width(), height()); }
inline CGRect cgRect() const { return CGRectMake({0, 0}, cgSize()); }
inline CGFloat cgWidth() const { return static_cast<CGFloat>(width()); }
inline CGFloat cgHeight() const { return static_cast<CGFloat>(height()); }
inline CGFloat cgSpectrum() const { return static_cast<CGFloat>(spectrum()); }

CGContextRef cgContext(CGColorSpaceRef colorspace) const {
    NSLog(@"");
    NSLog(@"BYTES PER ROW: %i", rowbytes());
    NSLog(@"SPECTRUM: %i", _spectrum);
    NSLog(@"SIZEOF(T): %i", (int)sizeof(T));
    NSLog(@"WxH: %ix%i", _width, _height);
    CGBitmapInfo bitmapInfo = isFloat() ? kCGBitmapFloatComponents : kCGBitmapByteOrderDefault;
    CGContextRef ctx = CGBitmapContextCreate(NULL,
        cgWidth(), cgHeight(),
        itemsize(), rowbytes(),
        colorspace,
        kCGImageAlphaNone | bitmapInfo);
    
    
    return ctx;
}

CGImageRef cgImageRef(CGContextRef context) const {
    return CGBitmapContextCreateImage(context);
}

NSURL *pathToURL(const char *path) {
    NSString *pth = [[NSString stringWithUTF8String:path] autorelease];
    return [[NSURL alloc] initFileURLWithPath:pth];
}
NSURL *pathToURL(NSString *pth) {
    return [[NSURL alloc] initFileURLWithPath:pth];
}

CFURLRef pathToURLRef(const char *path) {
    @autoreleasepool {
        return (CFURLRef)pathToURL(path);
    }
}
CFURLRef pathToURLRef(NSString *pth) {
    @autoreleasepool {
        return (CFURLRef)pathToURL(pth);
    }
}
