
#include "structcode.hpp"
using namespace std;

namespace structcode {
    
    /// static initializers
    const map<string, string> structcodemaps::byteorder = structcodemaps::init_byteorder();
    const map<string, string> structcodemaps::native = structcodemaps::init_native();
    const map<string, string> structcodemaps::standard = structcodemaps::init_standard();
    
    vector<int> parse_shape(string shapecode) {
        string segment;
        vector<int> shape_elems;
        while (shapecode.find(",", 0) != string::npos) {
            size_t pos = shapecode.find(",", 0);
            segment = shapecode.substr(0, pos);
            shapecode.erase(0, pos+1);
            shape_elems.push_back(stoi(segment));
        }
        shape_elems.push_back(stoi(shapecode));
        return shape_elems;
    }

    vector<pair<string, string>> parse(string structcode, bool toplevel) {
        vector<string> tokens;
        vector<pair<string, string>> fields;
        field_namer field_names;
        
        string byteorder = "@";
        size_t itemsize = 1;
        vector<int> shape = {0};
        const vector<int> noshape = shape;
        
        while (true) {
            if (structcode.size() == 0) { break; }
            switch (structcode[0]) {
                case '{': {
                    structcode.erase(0, 1);
                    int pos = 1;
                    size_t siz;
                    for (siz = 0; pos && (siz != structcode.size()); ++siz) {
                        if (structcode[siz] == '{') { ++pos; }
                        if (structcode[siz] == '}') { --pos; }
                    }
                    if (pos) { break; } /// too many open-brackets
                    string temp = structcode.substr(0, siz-1);
                    vector<pair<string, string>> pairvec;
                    pairvec = parse(temp, toplevel=false);
                    structcode.erase(0, siz+1);
                    for (size_t idx = 0; idx < pairvec.size(); ++idx) {
                        fields.push_back(pairvec[idx]);
                    }
                }
                break;
                case '(': {
                    structcode.erase(0, 1);
                    int pos = 1;
                    size_t siz;
                    for (siz = 0; pos && (siz != structcode.size()); ++siz) {
                        if (structcode[siz] == '(') { ++pos; }
                        if (structcode[siz] == ')') { --pos; }
                    }
                    if (pos) { break; } /// too many open-parens
                    string shapestr = structcode.substr(0, siz-1);
                    shape = parse_shape(shapestr);
                    structcode.erase(0, siz);
                }
                break;
                case '*':
                {
                    /// SECRET NOOp
                    structcode.erase(0, 1);
                    size_t pos = structcode.find("*", 0);
                    string explicit_name = structcode.substr(0, pos);
                    field_names.add(explicit_name);
                    structcode.erase(0, pos+1);
                }
                break;
                case '@':
                case '=':
                case '<':
                case '>':
                case '^':
                case '!':
                {
                    try {
                        byteorder = structcodemaps::byteorder.at(structcode.substr(0, 1));
                    } catch (const out_of_range &err) {
                        cerr    << ">>> Byte order symbol not found: "
                                << structcode.substr(0, 1) << "\n>>> Exception message: "
                                << err.what() << "\n";
                    }
                    structcode.erase(0, 1);
                    tokens.push_back(byteorder);
                    fields.push_back(make_pair("__byteorder__", byteorder));
                }    
                break;
                case ' ':
                {
                    /// NOP
                    structcode.erase(0, 1);
                }
                break;
                case '0':
                case '1':
                case '2':
                case '3':
                case '4':
                case '5':
                case '6':
                case '7':
                case '8':
                case '9':
                {
                    size_t siz;
                    char digit;
                    int still_digits = 1;
                    for (siz = 0; still_digits && (siz < structcode.size()); siz++) {
                        digit = structcode.c_str()[siz];
                        still_digits = isdigit(digit) && digit != '(';
                    }
                    string numstr = string(structcode.substr(0, siz));
                    itemsize = static_cast<size_t>(stol(numstr));
                    structcode.erase(0, siz);
                    if (!isdigit(numstr.back())) {
                        structcode = string(&numstr.back()) + structcode;
                    }
                }
                break;
                default:
                size_t codelen = 1;
                string code = "";
                string name = "";
                string dtypechar = "";
                
                if (structcode.substr(0, codelen) == "Z") {
                    /// add next character
                    codelen++;
                }
                
                code += string(structcode.substr(0, codelen));
                structcode.erase(0, codelen);
                
                /// field name
                if (structcode.substr(0, 1) == ":") {
                    structcode.erase(0, 1);
                    size_t pos = structcode.find(":", 0);
                    name = structcode.substr(0, pos);
                    field_names.add(name);
                    structcode.erase(0, pos+1);
                }
            
                name = name.size() ? name : field_names();
            
                if (byteorder == "@" || byteorder == "^") {
                    try {
                        dtypechar = structcodemaps::native.at(code);
                    } catch (const out_of_range &err) {
                        cerr    << ">>> Native structcode symbol not found: "
                                << code << "\n>>> Exception message: "
                                << err.what() << "\n";
                        try {
                            dtypechar = structcodemaps::standard.at(code);
                        } catch (const out_of_range &err) {
                            cerr    << ">>> Subsequent standard symbol lookup failed: "
                                    << code << "\n>>> Exception message: "
                                    << err.what() << "\n";
                        }
                        break;
                     }
                } else {
                    try {
                        dtypechar = structcodemaps::standard.at(code);
                    } catch (const out_of_range &err) {
                        cerr    << ">>> Standard structcode symbol not found: "
                                << code << "\n>>> Exception message: "
                                << err.what() << "\n";
                        try {
                            dtypechar = structcodemaps::native.at(code);
                        } catch (const out_of_range &err) {
                            cerr    << ">>> Subsequent native structcode symbol lookup failed: "
                                    << code << "\n>>> Exception message: "
                                    << err.what() << "\n";
                        }
                        break;
                    }
                }
                
                const char last = dtypechar.back();
                if (last == 'U' || last == 'S' || last == 'V') {
                    if (itemsize > 1) {
                        ostringstream outstream;
                        outstream << itemsize;
                        dtypechar += outstream.str();
                    }
                    itemsize = 1;
                }
                
                if (shape != noshape) {
                    ostringstream outstream;
                    outstream << "(";
                    for (auto shape_elem = begin(shape); shape_elem != end(shape); ++shape_elem) {
                        outstream << *shape_elem;
                        if (shape_elem + 1 != end(shape)) {
                            outstream << ", ";
                        }
                    }
                    outstream << ")";
                    dtypechar = outstream.str() + dtypechar;
                } else if (itemsize > 1) {
                    dtypechar = to_string(itemsize) + dtypechar;
                }
                
                fields.push_back(make_pair(name, dtypechar));
                tokens.push_back(dtypechar);
                
                //byteorder = "@";
                itemsize = 1;
                shape = noshape;
            
                break;
            }
        }
        
        return fields;
    }

} /// namespace structcode
