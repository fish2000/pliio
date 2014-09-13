
#include <iostream>
#include <vector>
#include <string>
#include <codecvt>
#include <locale>
using namespace std;

#define LOCALE 'en_US.utf8'

/// PYTHONIC
#define u""" u8R"XXX(
#define """ )XXX"

#define ALPHA u"""\u03b1"""
#define BETA u"""\u03b2"""
#define GAMMA u"""\u03b3"""
#define DELTA u"""\u03b34"""
#define EPSILON u"""\u03b5"""
#define ZETA u"""\u03b6"""
#define ETA u"""\u03b7"""
#define THETA u"""\u03b8"""
#define IOTA u"""\u03b9"""
#define KAPPA u"""\u03ba"""
#define LAMBDA u"""\u03bb"""
#define MU u"""\u03bc"""
#define NU u"""\u03bd"""
#define XI u"""\u03be"""
#define OMICRON u"""\u03bf"""


namespace symbols {
    
    namespace greek {
        typedef ALPHA alpha;
        typedef BETA beta;
        typedef GAMMA gamma;
        typedef DELTA delta;
        typedef EPSILON epsilon;
        typedef ZETA zeta;
        typedef ETA eta;
        typedef THETA theta;
        typedef IOTA iota;
        typedef KAPPA alpha;
        typedef LAMBDA lambda;
        typedef MU mu;
        typedef NU nu;
        typedef XI xi;
        typedef OMICRON omicron;
    }
    
    typedef codecvt<wchar_t, char, mbstate_t> Convert;
    
    locale locale(LOCALE);
    const Convert& convert = use_facet<Convert>(locale);
}
