
#ifndef PyImage_INTEGRALS_H
#define PyImage_INTEGRALS_H

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

#endif /// PyImage_INTEGRALS_H