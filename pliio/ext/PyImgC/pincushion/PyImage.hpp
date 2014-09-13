
#ifndef PyImage_PYIMAGE_H
#define PyImage_PYIMAGE_H

#include <iostream>
#include <vector>
#include <string>
#include <cstdarg>
#include "symbols.hpp"

#include "../numpypp/numpy.hpp"
#include "../numpypp/array.hpp"
#include "../numpypp/utils.hpp"

using namespace std;
using namespace symbols;

struct PyImageTypes {
    typedef char pc;
    typedef short ps;
    typedef int pi;
    typedef long pl;
    typedef long long pll;
    
    typedef unsigned char pC;
    typedef unsigned short pS;
    typedef unsigned int pI;
    typedef unsigned long pL;
    typedef unsigned long long pLL;
    
    typedef float pf;
    typedef double pd;
    typedef long double pdl;
    
    typedef static constexpr CCT;
    typedef static constexpr int CCi;
    typedef static constexpr char CCc;
    typedef static constexpr unsigned int CCI;
    typedef static constexpr unsigned char CCC;
};

struct PyImage_SubBase {
    virtual ~PyImage_SubBase() {};
};

template <typename pT, unsigned int NDIMS=3, unsigned int COLORDEPTH=4>
struct PyImage_ModeBase : virtual public PyImageTypes, virtual public PyImage_SubBase {
    typedef pT value_type;
    typedef sizeof(pT) value_size;
    
    CCT pT min = numeric_limits<pT>::min();
    CCT pT max = numeric_limits<pT>::max();
    CCI ndims = integral_constant<CCI, NDIMS>::value;
    CCI depth = integral_constant<CCI, COLORDEPTH>::value;
    CCc channel_characters[depth] = { greek::alpha, greek::beta, greek::lambda, greek::delta };
    
    virtual pI[ndims] shape_for_image_dims(pI width, pI height, ...) {
        pI out[ndims];
        out[0] = width; out[1] = height; out[2] = depth;
        if (ndims > 3) {
            va_list higher_orders;
            va_start(higher_orders, height);
            for (int idx = 3; idx < height; idx++) {
                out[idx] = va_arg(higher_orders, pI);
            };
            va_end(higher_orders);
        }
        return out;
    }

    /// make these channel type shits into ENUM CLASSES dogg
    vector<tuple<pT>> channels(NDIMS);
    virtual PyImage_ModeBase() {
        channels.push_back(make_tuple('r', ));
    }
};

template <typename pT>
struct PyImage_ColorBase<pT, 3, 3> : public TBImage_ModelBase<pT, 3, 3>{};

class PyImage {

public:
    PyImage() {}
    PyImage(PyImage&) {} /// or whatever -- 'copy costructor'
    PyImage(PyImage&&&) {} /// I really forget what here -- 'move constructor'
    
protected:
    /// stubs from _imaging.c and the like
    Py_buffer *getBuffer();
    static int getBands(string mode_string); /// I DUNOOOOO
    PyObject *asList(PyObject *arg, int* length, const char* wrong_length, int type); /// ALSO THIS I DUNNO
    static inline PyObject *getPixel(PyImage im, ImagingAccess access, int x, int y); /// Take a GOOD LOOK at this shit
    static char *getInk(PyObject *color, PyImage im, char* ink);
    
    //// these are all, like, wrappers --
    //// in PIL's source for _imaging.c, they all fly under
    /// a "FACTORIES" banner
    static PyObject *_fill(SELF+ARGS);
    static PyObject *_new(SELF+ARGS);
    static PyObject *_new_array(SELF+ARGS); /// ??! as in 'new from array' or 'make new fucking array'??
    static PyObject *_new_block(SELF+ARGS);
    static PyObject *_get_count(SELF+ARGS);
    static PyObject *_linear_gradient(SELF+ARGS);
    static PyObject *_radial_gradient(SELF+ARGS);
    static PyObject *_open_ppm(SELF+ARGS); /// file loader -- probably I don't give a fuck most likely
    static PyObject *_alpha_composite(SELF+ARGS);
    static PyObject *_blend(SELF+ARGS);
    static PyObject *_get_count(SELF+ARGS);
    
    /// ... whereas these that follow are all "METHODS"
    /// unlike the FACTORIES above (all of whom merely delegate out their calls,
    /// with little exception or embellishment) these have their own inline implemenations
    static PyObject *_convert(SELF+ARGS);
    static PyObject *_convert2(SELF+ARGS); /// what
    static PyObject *_convert_matrix(SELF+ARGS);
    static PyObject *_copy(SELF+ARGS);
    static PyObject *_copy2(SELF+ARGS);
    static PyObject *_crop(SELF+ARGS);
    static PyObject *_expand(SELF+ARGS);
    static PyObject *_filter(SELF+ARGS);
    static PyObject *_gaussian_blur(SELF+ARGS);
    static PyObject *_getpalette(SELF+ARGS);
    static PyObject *_getpalettemode(SELF+ARGS);
    static PyObject *_getxy(SELF+ARGS);
    static PyObject *_getpixel(SELF+ARGS);
    static PyObject *_gethistogram(SELF+ARGS); /// NB. NEED STRUCT TYPE FOR THIS
    static PyObject *_modefilter(SELF+ARGS);
    static PyObject *_offset(SELF+ARGS);
    static PyObject *_paste(SELF+ARGS);
    static PyObject *_point(SELF+ARGS);
    static PyObject *_point_transform(SELF+ARGS);
    static PyObject *_putdata(SELF+ARGS);
    static PyObject *_quantize(SELF+ARGS); /// DONT FORGET NEUQUANT!
    static PyObject *_putpalette(SELF+ARGS);
    static PyObject *_putpalettealpha(SELF+ARGS);
    static PyObject *_putpalettealphas(SELF+ARGS);
    static PyObject *_putpixel(SELF+ARGS);
    static PyObject *_rankfilter(SELF+ARGS);
    static PyObject *_rotate(SELF+ARGS);
    
    static PyObject *im_setmode(SELF+ARGS);
    static PyObject *_stretch(SELF+ARGS);
    static PyObject *_transform2(SELF+ARGS);
    static PyObject *_transpose(SELF+ARGS);
    static PyObject *_unsharp_mask(SELF+ARGS);

    static PyObject *_isblock(SELF+ARGS);
    static PyObject *_getbbox(SELF+ARGS);
    static PyObject *_getcolors(SELF+ARGS);
    static PyObject *_getextrema(SELF+ARGS); /// wtf does this mean
    static PyObject *_getprojection(SELF+ARGS); /// wtf does this mean too
    
    static PyObject *_getband(SELF+ARGS);
    static PyObject *_fillband(SELF+ARGS);
    static PyObject *_putband(SELF+ARGS);
    
    static PyObject *_chop_invert(SELF+ARGS);
    static PyObject *_chop_lighter(SELF+ARGS);
    static PyObject *_chop_darker(SELF+ARGS);
    static PyObject *_chop_multiply(SELF+ARGS);
    static PyObject *_chop_screen(SELF+ARGS);
    static PyObject *_chop_add(SELF+ARGS);
    static PyObject *_chop_subtract(SELF+ARGS);
    static PyObject *_chop_and(SELF+ARGS);
    static PyObject *_chop_or(SELF+ARGS);
    static PyObject *_chop_xor(SELF+ARGS);
    static PyObject *_chop_add_modulo(SELF+ARGS);
    static PyObject *_chop_subtract_modulo(SELF+ARGS);
    
    /// WITH IMAGEDRAW (he says, can you believe it)
    static PyObject *_font_new(SELF+ARGS);
    static void _font_dealloc(ImagingFontObject*);
    static inline int textwidth(self, cont unsigned char *text);
    static PyObject *_font_getmask(SELF+ARGS);
    /*
     * actually FUCK the rest of these, use PD's stuff
     */
    
    static PyObject *pixel_access_new(SELF+ARGS);
    static void pixel_access_dealloc(self);
    /// these next two used in:
    /// static PyMappingMethods pixel_access_as_mapping;
    /// (SEE BELOW)
    static PyObject *pixel_access_getitem(self, xy, val);
    static PyObject *pixel_access_setitem(self, xy);
    
    /* SPECIAL FX!!!! */
    static PyObject *_effect_mandelbrot(SELF+ARGS);
    static PyObject *_effect_noise(SELF+ARGS);
    static PyObject *_effect_spread(SELF+ARGS);
    
    /* UTILITIES */
    static PyObject *_crc32(SELF+ARGS);
    static PyObject *_getcodecstatus(SELF+ARGS); /// ?
    static PyObject *_save_ppm(SELF+ARGS); /// only WITH_DEBUG
    
    /* GETTERS and (NO) SETTERS (ACTUALLY) */
    static PyObject *_getattr_mode(SELF+CLOSURE);
    static PyObject *_getattr_size(SELF+CLOSURE);
    static PyObject *_getattr_bands(SELF+CLOSURE);
    static PyObject *_getattr_id(SELF+CLOSURE);
    static PyObject *_getattr_ptr(SELF+CLOSURE);
    
    /* sequence shit */
    static Py_ssize_t image_length(SELF);
    static Py_ssize_t image_item(SELF); /// returns a PIXEL
                                        /// ... LEARN TO NAME SHIT
    
    /* Types */
    static PyTypeObject Imaging_Type; /// {...}
    static PyTypeObject ImagingFont_Type; /// {...}
    static PyTypeObject ImagingDraw_Type; /// {...}
    
    static PyMappingMethods pixel_access_as_mapping; /// {...}
    static PyTypeObject PixelAccess_Type; /// {...}
    
    /* there are then a shitton of extern declarations,
     * for shit like codecs and Win32 hooks (ha) --
     * and a smattering of "experimental" bits
     * (notably PyImaging_Mapper and PyImaging_MapBuffer)
    
    
    
    
    
    
    
};



#endif /// PyImage_PYIMAGE_H