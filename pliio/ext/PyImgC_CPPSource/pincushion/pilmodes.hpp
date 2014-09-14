#ifndef PyImgC_PILMODES_H
#define PyImgC_PILMODES_H

#ifndef IMGC_DEBUG
#define IMGC_DEBUG 0
#endif

#if IMGC_DEBUG > 0
    #define IMGC_COUT(x) cout << x << "\n"
    #define IMGC_CERR(x) cerr << x << "\n"
#else
    #define IMGC_COUT(x)
    #define IMGC_CERR(x)
#endif

#include <cstdio>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>
#include <map>
using namespace std;

namespace pilmode {

    struct pilmodemaps {
        static map<string, string> init_modes() {
            static map<string, string> _modes = {
                
            };
            return _modes;
        }
    };
    
    const map<string, string> pilmodemaps::modes = pilmodemaps::init_modes();
}
    













#endif /// PyImgC_PILMODES_H