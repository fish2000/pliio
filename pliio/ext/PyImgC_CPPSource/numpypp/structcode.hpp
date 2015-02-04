#ifndef PyImgC_STRUCTCODE_H
#define PyImgC_STRUCTCODE_H

#include <cstdio>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>
#include <map>
using namespace std;

namespace structcode {

    struct structcodemaps {
        
        static map<string, string> init_byteorder() {
            map<string, string> _byteorder_map = {
                {"@", "="},
                {"=", "="},
                {"<", "<"},
                {">", ">"},
                {"^", "="},
                {"!", ">"},
            };
            return _byteorder_map;
        }
        
        static map<string, string> init_native() {
            map<string, string> _native_map = {
                {"?", "?"},
                {"b", "b"},
                {"B", "B"},
                {"h", "h"},
                {"H", "H"},
                {"i", "i"},
                {"I", "I"},
                {"l", "l"},
                {"L", "L"},
                {"q", "q"},
                {"Q", "Q"},
                {"e", "e"},
                {"f", "f"},
                {"d", "d"},
                {"g", "g"}, 
                {"Zf", "F"},
                {"Zd", "D"},
                {"Zg", "G"},
                {"s", "S"},
                {"w", "U"},
                {"O", "O"},
                {"x", "V"}, /// padding
            };
            return _native_map;
        }
        
        static map<string, string> init_standard() {
            map<string, string> _standard_map = {
                {"?", "?"},
                {"b", "b"},
                {"B", "B"},
                {"h", "i2"},
                {"H", "u2"},
                {"i", "i4"},
                {"I", "u4"},
                {"l", "i4"},
                {"L", "u4"},
                {"q", "i8"},
                {"Q", "u8"},
                {"e", "f2"},
                {"f", "f"},
                {"d", "d"},
                {"Zf", "F"},
                {"Zd", "D"},
                {"s", "S"},
                {"w", "U"},
                {"O", "O"},
                {"x", "V"}, /// padding
            };
            return _standard_map;
        }
        
        static const map<string, string> byteorder;
        static const map<string, string> native;
        static const map<string, string> standard;
    };

    //const map<string, string> structcodemaps::byteorder = structcodemaps::init_byteorder();
    //const map<string, string> structcodemaps::native = structcodemaps::init_native();
    //const map<string, string> structcodemaps::standard = structcodemaps::init_standard();

    struct field_namer {
        int idx;
        vector<string> field_name_vector;
        field_namer():idx(0) {}
        int next() { return idx++; }
        void add(string field_name) { field_name_vector.push_back(field_name); }
        bool has(string field_name) {
            for (auto fn = begin(field_name_vector); fn != end(field_name_vector); ++fn) {
                if (string(*fn) == field_name) {
                    return true;
                }
            }
            return false;
        }
        string operator()() {
            char str[5];
            while (true) {
                sprintf(str, "f%i", next());
                string dummy_name = string(str);
                if (!has(dummy_name)) {
                    add(dummy_name);
                    return dummy_name;
                }
            }
        }
    };

    vector<int> parse_shape(string shapecode);
    vector<pair<string, string>> parse(string structcode, bool toplevel=true);

} /// namespace structcode

#endif