#ifndef PyImgC_TYPECODE_H
#define PyImgC_TYPECODE_H

#ifndef IMGC_DEBUG
#define IMGC_DEBUG 0
#endif

#include <Python.h>
#include <structmember.h>

#include <cstdio>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>
#include <map>

#ifndef NPY_NO_DEPRECATED_API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#endif

#include <numpy/arrayobject.h>
#include <numpy/ndarraytypes.h>
#include "../PyImgC_Constants.h"
using namespace std;

namespace typecode {
    
    typedef integral_constant<NPY_TYPES, NPY_BOOL> ENUM_NPY_BOOL;
    typedef integral_constant<NPY_TYPES, NPY_BYTE> ENUM_NPY_BYTE;
    typedef integral_constant<NPY_TYPES, NPY_HALF> ENUM_NPY_HALF;
    typedef integral_constant<NPY_TYPES, NPY_SHORT> ENUM_NPY_SHORT;
    typedef integral_constant<NPY_TYPES, NPY_INT> ENUM_NPY_INT;
    typedef integral_constant<NPY_TYPES, NPY_LONG> ENUM_NPY_LONG;
    typedef integral_constant<NPY_TYPES, NPY_LONGLONG> ENUM_NPY_LONGLONG;
    typedef integral_constant<NPY_TYPES, NPY_UBYTE> ENUM_NPY_UBYTE;
    typedef integral_constant<NPY_TYPES, NPY_USHORT> ENUM_NPY_USHORT;
    typedef integral_constant<NPY_TYPES, NPY_UINT> ENUM_NPY_UINT;
    typedef integral_constant<NPY_TYPES, NPY_ULONG> ENUM_NPY_ULONG;
    typedef integral_constant<NPY_TYPES, NPY_ULONGLONG> ENUM_NPY_ULONGLONG;
    typedef integral_constant<NPY_TYPES, NPY_CFLOAT> ENUM_NPY_CFLOAT;
    typedef integral_constant<NPY_TYPES, NPY_CDOUBLE> ENUM_NPY_CDOUBLE;
    typedef integral_constant<NPY_TYPES, NPY_FLOAT> ENUM_NPY_FLOAT;
    typedef integral_constant<NPY_TYPES, NPY_DOUBLE> ENUM_NPY_DOUBLE;
    typedef integral_constant<NPY_TYPES, NPY_CLONGDOUBLE> ENUM_NPY_CLONGDOUBLE;
    typedef integral_constant<NPY_TYPES, NPY_LONGDOUBLE> ENUM_NPY_LONGDOUBLE;

    struct typecodemaps {
    
        static map<int, NPY_TYPES> init_integral_map() {
            map<int, NPY_TYPES> _integral_map = {
                { NPY_BOOL, ENUM_NPY_BOOL::value }, 
                { NPY_BYTE, ENUM_NPY_BYTE::value }, 
                { NPY_HALF, ENUM_NPY_HALF::value }, 
                { NPY_SHORT, ENUM_NPY_SHORT::value }, 
                { NPY_INT, ENUM_NPY_INT::value }, 
                { NPY_LONG, ENUM_NPY_LONG::value }, 
                { NPY_LONGLONG, ENUM_NPY_LONGLONG::value }, 
                { NPY_UBYTE, ENUM_NPY_UBYTE::value }, 
                { NPY_USHORT, ENUM_NPY_USHORT::value }, 
                { NPY_UINT, ENUM_NPY_UINT::value }, 
                { NPY_ULONG, ENUM_NPY_ULONG::value }, 
                { NPY_ULONGLONG, ENUM_NPY_ULONGLONG::value }, 
                { NPY_CFLOAT, ENUM_NPY_CFLOAT::value }, 
                { NPY_CDOUBLE, ENUM_NPY_CDOUBLE::value }, 
                { NPY_FLOAT, ENUM_NPY_FLOAT::value }, 
                { NPY_DOUBLE, ENUM_NPY_DOUBLE::value }, 
                { NPY_CLONGDOUBLE, ENUM_NPY_CLONGDOUBLE::value }, 
                { NPY_LONGDOUBLE, ENUM_NPY_LONGDOUBLE::value }
            };
            return _integral_map;
        }
        
        static map<NPY_TYPES, NPY_TYPECHAR> init_typecode_character_map() {
            map<NPY_TYPES, NPY_TYPECHAR> _typecode_character_map = {
                { NPY_BOOL, NPY_BOOLLTR },
                { NPY_BYTE, NPY_BYTELTR },
                { NPY_UBYTE, NPY_UBYTELTR },
                { NPY_SHORT, NPY_SHORTLTR },
                { NPY_USHORT, NPY_USHORTLTR },
                { NPY_INT, NPY_INTLTR },
                { NPY_UINT, NPY_UINTLTR },
                { NPY_LONG, NPY_LONGLTR },
                { NPY_ULONG, NPY_ULONGLTR },
                { NPY_LONGLONG, NPY_LONGLONGLTR },
                { NPY_ULONGLONG, NPY_ULONGLONGLTR },
                { NPY_FLOAT, NPY_FLOATLTR },
                { NPY_DOUBLE, NPY_DOUBLELTR },
                { NPY_LONGDOUBLE, NPY_LONGDOUBLELTR },
                { NPY_CFLOAT, NPY_CFLOATLTR },
                { NPY_CDOUBLE, NPY_CDOUBLELTR },
                { NPY_CLONGDOUBLE, NPY_CLONGDOUBLELTR },
                { NPY_OBJECT, NPY_OBJECTLTR },
                { NPY_STRING, NPY_STRINGLTR },
                { NPY_UNICODE, NPY_UNICODELTR },
                { NPY_VOID, NPY_VOIDLTR },
                { NPY_DATETIME, NPY_DATETIMELTR },
                { NPY_HALF, NPY_HALFLTR },
                { NPY_TIMEDELTA, NPY_TIMEDELTALTR },
                { NPY_CHAR, NPY_CHARLTR },
                { NPY_USERDEF, NPY_BOOLLTR }, /// bah
            };
            return _typecode_character_map;
        }
        
        static map<NPY_TYPES, string> init_typecode_literal_map() {
            map<NPY_TYPES, string> _typecode_literal_map = {
                { NPY_BOOL, "NPY_BOOL" },
                { NPY_BYTE, "NPY_BYTE" },
                { NPY_UBYTE, "NPY_UBYTE" },
                { NPY_SHORT, "NPY_SHORT" },
                { NPY_USHORT, "NPY_USHORT" },
                { NPY_INT, "NPY_INT" },
                { NPY_UINT, "NPY_UINT" },
                { NPY_LONG, "NPY_LONG" },
                { NPY_ULONG, "NPY_ULONG" },
                { NPY_LONGLONG, "NPY_LONGLONG" },
                { NPY_ULONGLONG, "NPY_ULONGLONG" },
                { NPY_FLOAT, "NPY_FLOAT" },
                { NPY_DOUBLE, "NPY_DOUBLE" },
                { NPY_LONGDOUBLE, "NPY_LONGDOUBLE" },
                { NPY_CFLOAT, "NPY_CFLOAT" },
                { NPY_CDOUBLE, "NPY_CDOUBLE" },
                { NPY_CLONGDOUBLE, "NPY_CLONGDOUBLE" },
                { NPY_OBJECT, "NPY_OBJECT" },
                { NPY_STRING, "NPY_STRING" },
                { NPY_UNICODE, "NPY_UNICODE" },
                { NPY_VOID, "NPY_VOID" },
                { NPY_DATETIME, "NPY_DATETIME" },
                { NPY_HALF, "NPY_HALF" },
                { NPY_TIMEDELTA, "NPY_TIMEDELTA" },
                { NPY_CHAR, "NPY_CHAR" },
                { NPY_USERDEF, "?" }
            };
            return _typecode_literal_map;
        }
        
        static const map<int, NPY_TYPES> integral;
        static const map<NPY_TYPES, NPY_TYPECHAR> character;
        static const map<NPY_TYPES, string> literal;
    };

    const map<int, NPY_TYPES> typecodemaps::integral = typecodemaps::init_integral_map();
    const map<NPY_TYPES, NPY_TYPECHAR> typecodemaps::character = typecodemaps::init_typecode_character_map();
    const map<NPY_TYPES, string> typecodemaps::literal = typecodemaps::init_typecode_literal_map();
    
    static inline NPY_TYPECHAR typechar(NPY_TYPES typecode) {
        try {
            return typecodemaps::character.at(typecode);
        } catch (const out_of_range &err) {
            cerr    << ">>> Type character not found for typecode: "
                    << typecode << "\n>>> Exception message: "
                    << err.what() << "\n";
            return typecodemaps::character.at(NPY_USERDEF);
        }
    }
    static inline NPY_TYPECHAR typechar(unsigned int typecode) {
        return typecode::typechar(static_cast<NPY_TYPES>(typecode));
    }
    
    static inline string typecode_name(NPY_TYPES typecode) {
        try {
            return typecodemaps::literal.at(typecode);
        } catch (const out_of_range &err) {
            cerr    << ">>> Typecode literal not found for typecode: "
                    << typecode << "\n>>> Exception message: "
                    << err.what() << "\n";
            return typecodemaps::literal.at(NPY_USERDEF);
        }
    }
    static inline string typecode_name(unsigned int typecode) {
        return typecode::typecode_name(static_cast<NPY_TYPES>(typecode));
    }
}


#endif