
#import <Cocoa/Cocoa.h>
#import "ICC.h"

@interface Profile : NSObject
{
    ColorSyncProfileRef     mRef;
    CGColorSpaceRef         mColorspace;
    CFURLRef                mURL;
    icProfileClassSignature mClass;
    icColorSpaceSignature   mSpace;
    CFStringRef             mName;
    CFStringRef             mPath;
}

+ (NSArray*) arrayOfAllProfiles;
+ (NSArray*) arrayOfAllProfilesWithSpace:(icColorSpaceSignature)space;

+ (Profile*) profileDefaultRGB;
+ (Profile*) profileDefaultGray;
+ (Profile*) profileDefaultCMYK;

+ (Profile*) profileWithIterateData:(CFDictionaryRef) data;
- (Profile*) initWithIterateData:(CFDictionaryRef) data;
+ (Profile*) profileWithPath:(CFStringRef) path;
- (Profile*) initWithCFPath:(CFStringRef) path;
+ (Profile*) profileWithSRGB;
- (Profile*) initWithSRGB;
//+ (Profile*) profileWithLinearRGB;
//- (Profile*) initWithLinearRGB;

- (ColorSyncProfileRef) ref;
- (CFURLRef) url;
- (icProfileClassSignature) classType;
- (icColorSpaceSignature) spaceType;
- (NSString*) description;
- (NSString*) path;
- (CGColorSpaceRef) colorspace;

- (id) valueForUndefinedKey:(NSString*)key;

@end

CGImageRef CGImageCreateCopyWithDefaultSpace (CGImageRef image);
