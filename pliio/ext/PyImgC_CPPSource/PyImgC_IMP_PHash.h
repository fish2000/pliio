
#ifndef PyImgC_IMP_PHASH_H
#define PyImgC_IMP_PHASH_H

#include <iostream>
#include <string>
#include <limits.h>
#include <math.h>
#include <sys/types.h>
#include <stdlib.h>
#define __STDC_CONSTANT_MACROS
#include <stdint.h>
#include <Python.h>

#include "PyImgC_Constants.h"
#include "PyImgC_PyCImage.h"
#include "PyImgC_IMP_Utils.h"
using namespace std;

#define SQRT_TWO 1.4142135623730950488016887242097
#define ROUNDING_FACTOR(x) (((x) >= 0) ? 0.5 : -0.5) 
//#define max(a, b) (((a) > (b)) ? (a) : (b))

typedef struct ph_projections {
    CImg<uint8_t> *R;
    int *nb_pix_perline;
    int size;
} Projections;

typedef struct ph_feature_vector {
    double *features;
    int size;
} Features;

typedef struct ph_digest {
    char *id;
    uint8_t *coeffs;
    int size;
} Digest;

int ph_radon_projections(const CImg<uint8_t> &img, int N, Projections &projs);
int ph_feature_vector(const Projections &projs, Features &fv);

int ph_dct(const Features &fv, Digest &digest);
int ph_crosscorr(const Digest &x, const Digest &y, double &pcc, double threshold=0.90);
int ph_image_digest(const CImg<uint8_t> &img, double sigma, double gamma,
                     Digest &digest, int N=180);

int ph_compare_images(const CImg<uint8_t> &imA, const CImg<uint8_t> &imB,
                       double &pcc,
                       double sigma=3.5, double gamma=1.0,
                       int N=180, double threshold=0.90);

CImg<float> *ph_dct_matrix(const int N);

unsigned long long ph_dct_imagehash(CImg<uint8_t> src);

int ph_hamming_distance(const unsigned long long hash1, const unsigned long long hash2);

CImg<float> *ph_mh_kernel(float alpha, float level);

uint8_t *ph_mh_imagehash(CImg<uint8_t> src, float alpha=2.0f, float lvl=1.0f);

int ph_bitcount8(uint8_t val);

double ph_hamming_distance2(uint8_t *hashA, uint8_t *hashB, int lenA=72, int lenB=72);

#endif /// PyImgC_IMP_PHASH_H