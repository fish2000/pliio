#define NILCODE '~'

template <IMGT, typename dT>
unique_ptr<CImage_SubBase> create() {
    //return unique_ptr<CImage_Type<dT>>(new T());
    return unique_ptr<CImage_Type<dT>>(new CImage_Type<dT>());
}

typedef std::map<unsigned int, unique_ptr<CImage_SubBase>(*)()> CImage_TypeMap;
static CImage_TypeMap *tmap;

struct CImage_FunctorType {
    static inline CImage_TypeMap *get_map() {
        if (!tmap) { tmap = new CImage_TypeMap(); }
        return tmap;
    }
};

template <typename dT>
static inline CImage_Type<dT> *CImage_NumpyConverter(unsigned int key) {
    // CImage_TypeMap::iterator it = CImage_FunctorType::get_map()->find(key);
    // if (it == CImage_FunctorType::get_map()->end()) {
        return new CImage_Type<dT>();
    // }
    // return dynamic_cast<CImage_Type<dT>*>(it->second());
}


//// UUUGGGGGGHHHHHHH
template <typename dT>
static inline CImage_Type<dT> *CImage_NumpyConverter(PyObject *pyarray) {
    return new CImage_Type<dT>(pyarray);
}

template <typename dT>
static inline unique_ptr<CImage_Type<dT>> CImage_TypePointer(PyObject *pyarray) {
    IMGC_COUT("> Calling CImage_TypePointer with pyarray: " << reinterpret_cast<PyTypeObject *>(PyObject_Type(pyarray))->tp_name);
    return unique_ptr<CImage_Type<dT>>(new CImage_Type<dT>(pyarray));
}
template <typename dT>
static inline unique_ptr<CImage_Type<dT>> CImage_TypePointer(CImg<dT> cimage) {
    IMGC_COUT("> Calling CImage_TypePointer with cimage: " << cimage.pixel_type());
    return unique_ptr<CImage_Type<dT>>(new CImage_Type<dT>(cimage));
}

template <NPY_TYPES, IMGT>
struct CImage_Functor : public CImage_FunctorType {};

/////////////////////////////////// AUTOGENERATED ///////////////////////////////////////
/////////////////////////////////// AUTOGENERATED ///////////////////////////////////////
/////////////////////////////////// AUTOGENERATED ///////////////////////////////////////

struct CImage_NPY_BOOL : public CImage_Type<bool> {
    const char structcode[2] = { '?', NILCODE };
    const unsigned int structcode_length = 2;
    const bool native = true;
    const bool complex = false;
    CImage_NPY_BOOL() {}
    CImage_Functor<NPY_BOOL, bool> reg();
};

struct CImage_NPY_BYTE : public CImage_Type<char> {
    const char structcode[2] = { 'b', NILCODE };
    const unsigned int structcode_length = 2;
    const bool native = true;
    const bool complex = false;
    CImage_NPY_BYTE() {}
    CImage_Functor<NPY_BYTE, char> reg();
};

struct CImage_NPY_HALF : public CImage_Type<npy_half> {
    const char structcode[2] = { 'e', NILCODE };
    const unsigned int structcode_length = 2;
    const bool native = false;
    const bool complex = false;
    CImage_NPY_HALF() {}
    CImage_Functor<NPY_HALF, npy_half> reg();
};

struct CImage_NPY_SHORT : public CImage_Type<short> {
    const char structcode[2] = { 'h', NILCODE };
    const unsigned int structcode_length = 2;
    const bool native = true;
    const bool complex = false;
    CImage_NPY_SHORT() {}
    CImage_Functor<NPY_SHORT, short> reg();
};

struct CImage_NPY_INT : public CImage_Type<int> {
    const char structcode[2] = { 'i', NILCODE };
    const unsigned int structcode_length = 2;
    const bool native = true;
    const bool complex = false;
    CImage_NPY_INT() {}
    CImage_Functor<NPY_INT, int> reg();
};

struct CImage_NPY_LONG : public CImage_Type<long> {
    const char structcode[2] = { 'l', NILCODE };
    const unsigned int structcode_length = 2;
    const bool native = true;
    const bool complex = false;
    CImage_NPY_LONG() {}
    CImage_Functor<NPY_LONG, long> reg();
};

struct CImage_NPY_LONGLONG : public CImage_Type<long long> {
    const char structcode[2] = { 'q', NILCODE };
    const unsigned int structcode_length = 2;
    const bool native = true;
    const bool complex = false;
    CImage_NPY_LONGLONG() {}
    CImage_Functor<NPY_LONGLONG, long long> reg();
};

struct CImage_NPY_UBYTE : public CImage_Type<unsigned char> {
    const char structcode[2] = { 'B', NILCODE };
    const unsigned int structcode_length = 2;
    const bool native = true;
    const bool complex = false;
    CImage_NPY_UBYTE() {}
    CImage_Functor<NPY_UBYTE, unsigned char> reg();
};

struct CImage_NPY_USHORT : public CImage_Type<unsigned short> {
    const char structcode[2] = { 'H', NILCODE };
    const unsigned int structcode_length = 2;
    const bool native = true;
    const bool complex = false;
    CImage_NPY_USHORT() {}
    CImage_Functor<NPY_USHORT, unsigned short> reg();
};

struct CImage_NPY_UINT : public CImage_Type<unsigned int> {
    const char structcode[2] = { 'I', NILCODE };
    const unsigned int structcode_length = 2;
    const bool native = true;
    const bool complex = false;
    CImage_NPY_UINT() {}
    CImage_Functor<NPY_UINT, unsigned int> reg();
};

struct CImage_NPY_ULONG : public CImage_Type<unsigned long> {
    const char structcode[2] = { 'L', NILCODE };
    const unsigned int structcode_length = 2;
    const bool native = true;
    const bool complex = false;
    CImage_NPY_ULONG() {}
    CImage_Functor<NPY_ULONG, unsigned long> reg();
};

struct CImage_NPY_ULONGLONG : public CImage_Type<unsigned long long> {
    const char structcode[2] = { 'Q', NILCODE };
    const unsigned int structcode_length = 2;
    const bool native = true;
    const bool complex = false;
    CImage_NPY_ULONGLONG() {}
    CImage_Functor<NPY_ULONGLONG, unsigned long long> reg();
};

struct CImage_NPY_CFLOAT : public CImage_Type<std::complex<float>> {
    const char structcode[2] = { 'f', NILCODE };
    const unsigned int structcode_length = 2;
    const bool native = false;
    const bool complex = true;
    CImage_NPY_CFLOAT() {}
    CImage_Functor<NPY_CFLOAT, std::complex<float>> reg();
};

struct CImage_NPY_CDOUBLE : public CImage_Type<std::complex<double>> {
    const char structcode[2] = { 'd', NILCODE };
    const unsigned int structcode_length = 2;
    const bool native = false;
    const bool complex = true;
    CImage_NPY_CDOUBLE() {}
    CImage_Functor<NPY_CDOUBLE, std::complex<double>> reg();
};

struct CImage_NPY_FLOAT : public CImage_Type<float> {
    const char structcode[2] = { 'f', NILCODE };
    const unsigned int structcode_length = 2;
    const bool native = false;
    const bool complex = false;
    CImage_NPY_FLOAT() {}
    CImage_Functor<NPY_FLOAT, float> reg();
};

struct CImage_NPY_DOUBLE : public CImage_Type<double> {
    const char structcode[2] = { 'd', NILCODE };
    const unsigned int structcode_length = 2;
    const bool native = false;
    const bool complex = false;
    CImage_NPY_DOUBLE() {}
    CImage_Functor<NPY_DOUBLE, double> reg();
};

struct CImage_NPY_CLONGDOUBLE : public CImage_Type<std::complex<long double>> {
    const char structcode[2] = { 'g', NILCODE };
    const unsigned int structcode_length = 2;
    const bool native = true;
    const bool complex = false;
    CImage_NPY_CLONGDOUBLE() {}
    CImage_Functor<NPY_CLONGDOUBLE, std::complex<long double>> reg();
};

struct CImage_NPY_LONGDOUBLE : public CImage_Type<std::complex<long double>> {
    const char structcode[2] = { 'g', NILCODE };
    const unsigned int structcode_length = 2;
    const bool native = true;
    const bool complex = true;
    CImage_NPY_LONGDOUBLE() {}
    CImage_Functor<NPY_LONGDOUBLE, std::complex<long double>> reg();
};

template <>
struct CImage_Functor<NPY_BOOL, bool> : public CImage_FunctorType {
    CImage_Functor<NPY_BOOL, bool>(unsigned int const& key) {
        get_map()->insert(make_pair(key, &create<CImage_Type<bool>, bool>));
    }
};

template <>
struct CImage_Functor<NPY_BYTE, char> : public CImage_FunctorType {
    CImage_Functor<NPY_BYTE, char>(unsigned int const& key) {
        get_map()->insert(make_pair(key, &create<CImage_Type<char>, char>));
    }
};

template <>
struct CImage_Functor<NPY_HALF, npy_half> : public CImage_FunctorType {
    CImage_Functor<NPY_HALF, npy_half>(unsigned int const& key) {
        get_map()->insert(make_pair(key, &create<CImage_Type<npy_half>, npy_half>));
    }
};

template <>
struct CImage_Functor<NPY_SHORT, short> : public CImage_FunctorType {
    CImage_Functor<NPY_SHORT, short>(unsigned int const& key) {
        get_map()->insert(make_pair(key, &create<CImage_Type<short>, short>));
    }
};

template <>
struct CImage_Functor<NPY_INT, int> : public CImage_FunctorType {
    CImage_Functor<NPY_INT, int>(unsigned int const& key) {
        get_map()->insert(make_pair(key, &create<CImage_Type<int>, int>));
    }
};

template <>
struct CImage_Functor<NPY_LONG, long> : public CImage_FunctorType {
    CImage_Functor<NPY_LONG, long>(unsigned int const& key) {
        get_map()->insert(make_pair(key, &create<CImage_Type<long>, long>));
    }
};

template <>
struct CImage_Functor<NPY_LONGLONG, long long> : public CImage_FunctorType {
    CImage_Functor<NPY_LONGLONG, long long>(unsigned int const& key) {
        get_map()->insert(make_pair(key, &create<CImage_Type<long long>, long long>));
    }
};

template <>
struct CImage_Functor<NPY_UBYTE, unsigned char> : public CImage_FunctorType {
    CImage_Functor<NPY_UBYTE, unsigned char>(unsigned int const& key) {
        get_map()->insert(make_pair(key, &create<CImage_Type<unsigned char>, unsigned char>));
    }
};

template <>
struct CImage_Functor<NPY_USHORT, unsigned short> : public CImage_FunctorType {
    CImage_Functor<NPY_USHORT, unsigned short>(unsigned int const& key) {
        get_map()->insert(make_pair(key, &create<CImage_Type<unsigned short>, unsigned short>));
    }
};

template <>
struct CImage_Functor<NPY_UINT, unsigned int> : public CImage_FunctorType {
    CImage_Functor<NPY_UINT, unsigned int>(unsigned int const& key) {
        get_map()->insert(make_pair(key, &create<CImage_Type<unsigned int>, unsigned int>));
    }
};

template <>
struct CImage_Functor<NPY_ULONG, unsigned long> : public CImage_FunctorType {
    CImage_Functor<NPY_ULONG, unsigned long>(unsigned int const& key) {
        get_map()->insert(make_pair(key, &create<CImage_Type<unsigned long>, unsigned long>));
    }
};

template <>
struct CImage_Functor<NPY_ULONGLONG, unsigned long long> : public CImage_FunctorType {
    CImage_Functor<NPY_ULONGLONG, unsigned long long>(unsigned int const& key) {
        get_map()->insert(make_pair(key, &create<CImage_Type<unsigned long long>, unsigned long long>));
    }
};

template <>
struct CImage_Functor<NPY_CFLOAT, std::complex<float>> : public CImage_FunctorType {
    CImage_Functor<NPY_CFLOAT, std::complex<float>>(unsigned int const& key) {
        get_map()->insert(make_pair(key, &create<CImage_Type<std::complex<float>>, std::complex<float>>));
    }
};

template <>
struct CImage_Functor<NPY_CDOUBLE, std::complex<double>> : public CImage_FunctorType {
    CImage_Functor<NPY_CDOUBLE, std::complex<double>>(unsigned int const& key) {
        get_map()->insert(make_pair(key, &create<CImage_Type<std::complex<double>>, std::complex<double>>));
    }
};

template <>
struct CImage_Functor<NPY_FLOAT, float> : public CImage_FunctorType {
    CImage_Functor<NPY_FLOAT, float>(unsigned int const& key) {
        get_map()->insert(make_pair(key, &create<CImage_Type<float>, float>));
    }
};

template <>
struct CImage_Functor<NPY_DOUBLE, double> : public CImage_FunctorType {
    CImage_Functor<NPY_DOUBLE, double>(unsigned int const& key) {
        get_map()->insert(make_pair(key, &create<CImage_Type<double>, double>));
    }
};

template <>
struct CImage_Functor<NPY_CLONGDOUBLE, std::complex<long double>> : public CImage_FunctorType {
    CImage_Functor<NPY_CLONGDOUBLE, std::complex<long double>>(unsigned int const& key) {
        get_map()->insert(make_pair(key, &create<CImage_Type<std::complex<long double>>, std::complex<long double>>));
    }
};

template <>
struct CImage_Functor<NPY_LONGDOUBLE, std::complex<long double>> : public CImage_FunctorType {
    CImage_Functor<NPY_LONGDOUBLE, std::complex<long double>>(unsigned int const& key) {
        get_map()->insert(make_pair(key, &create<CImage_Type<std::complex<long double>>, std::complex<long double>>));
    }
};

/////////////////////////////////// !AUTOGENERATED ///////////////////////////////////////
/////////////////////////////////// !AUTOGENERATED ///////////////////////////////////////
/////////////////////////////////// !AUTOGENERATED ///////////////////////////////////////

extern "C" void CImage_Register() {}
