
#ifndef PyHashTree_PYHASHTREE_IMP_DISTANCEFUNCTIONS_H
#define PyHashTree_PYHASHTREE_IMP_DISTANCEFUNCTIONS_H

static unsigned long long nbcalcs = 0;

/// hamming distance function
static float PyHashTree_DF_HammingDistance(MVPDP *ptA, MVPDP *ptB) {
    if (!ptA->data || !ptB->data || ptA->datalen != ptB->datalen) { return -1.0f; }

    uint64_t a = *((uint64_t *)ptA->data);
    uint64_t b = *((uint64_t *)ptB->data);

    uint64_t x = a ^ b;
    static const uint64_t m1  = 0x5555555555555555ULL;
    static const uint64_t m2  = 0x3333333333333333ULL;
    static const uint64_t h01 = 0x0101010101010101ULL;
    static const uint64_t m4  = 0x0f0f0f0f0f0f0f0fULL;
    x -= (x >> 1) & m1;
    x = (x & m2) + ((x >> 2) & m2);
    x = (x + (x >> 4)) & m4;

    float result = static_cast<float>((x * h01) >> 56);
    result = exp(result - 1);
    nbcalcs++;
    
    return result;
}

#endif /// PyHashTree_PYHASHTREE_IMP_DISTANCEFUNCTIONS_H