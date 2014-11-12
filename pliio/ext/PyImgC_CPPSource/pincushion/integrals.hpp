
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
    typedef static constexpr char CCl;
    typedef static constexpr unsigned int CCI;
    typedef static constexpr unsigned char CCC;
    typedef static constexpr unsigned long CCL;
};

struct PyImage_SubBase {
    virtual ~PyImage_SubBase() {};
};

template <unsigned int NDIMS=3>
struct PyImage_ShapeBase : virtual public PyImageTypes {
    CCI ndims = integral_constant<CCI, NDIMS>::value;
    pI dims[ndims];
    pI *operator()() { return dims; }
    pI size() const { return ndims; }
    ~PyImage_ShapeBase() {}
};

template <unsigned int NDIMS=3>
struct PyImage_Shape : virtual public PyImage_ShapeBase<NDIMS> {
    pI *operator[](pI idx) { return idx >= ndims ? nullptr : &dims[idx]; }
};

template <>
struct PyImage_Shape<1> : virtual public PyImage_ShapeBase<1> {
    PyImage_Shape(pI dim_alpha=0) {
        dims[0] = dim_alpha;
    }
};

template <>
struct PyImage_Shape<2> : virtual public PyImage_ShapeBase<2> {
    PyImage_Shape(pI dim_alpha=0, pI dim_beta=0) {
        dims[0] = dim_alpha;
        dims[1] = dim_beta;
    }
};

template <>
struct PyImage_Shape<3> : virtual public PyImage_ShapeBase<3> {
    PyImage_Shape(pI dim_alpha=0, pI dim_beta=0, pI dim_gamma=0) {
        dims[0] = dim_alpha;
        dims[1] = dim_beta;
        dims[2] = dim_gamma;
    }
};

template <>
struct PyImage_Shape<4> : virtual public PyImage_ShapeBase<4> {
    PyImage_Shape(pI dim_alpha=0, pI dim_beta=0, pI dim_gamma=0, pI dim_delta=4) {
        dims[0] = dim_alpha;
        dims[1] = dim_beta;
        dims[2] = dim_gamma;
        dims[3] = dim_delta;
    }
};


template <typename pT, unsigned int NDIMS=3, unsigned int COLORDEPTH=4>
struct PyImage_ModeBase : virtual public PyImageTypes, virtual public PyImage_SubBase {
    typedef pT value_type;
    typedef integral_constant<CCL, sizeof(pT)>::value value_size;
    
    CCT pT min = numeric_limits<pT>::min();
    CCT pT max = numeric_limits<pT>::max();
    CCI ndims = integral_constant<CCI, NDIMS>::value;
    CCI depth = integral_constant<CCI, COLORDEPTH>::value;
    CCI mindim = integral_constant<CCI, 1>::value;
    CCI mindepth = integral_constant<CCI, 1>::value;
    
    CCc channel_characters[depth] = { greek::alpha, greek::beta, greek::lambda, greek::delta };
    
    /// make these channel type shits into ENUM CLASSES dogg
    vector<tuple<CC, pT>> channels(NDIMS);
    virtual PyImage_ModeBase() {
        channels.push_back(make_tuple('r', ));
    }
    
};

template <typename pT>
struct PyImage_ColorBase<pT, 3, 3> : public TBImage_ModelBase<pT, 3, 3>{};

#endif /// PyImage_INTEGRALS_H