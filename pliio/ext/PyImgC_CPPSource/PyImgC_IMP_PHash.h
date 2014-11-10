
#ifndef PyImgC_IMP_PHASH_H
#define PyImgC_IMP_PHASH_H

#include <string>
#include "numpypp/utils.hpp"
using namespace std;


CImg<float> *ph_dct_matrix(const int N) {
    CImg<float> *ptr_matrix = new CImg<float>(N, N, 1, 1, 1 / sqrt((float)N));
    const float c1 = sqrt(2.0 / N);
    for (int x = 0; x < N; x++) {
        for (int y = 1; y < N; y++) {
            *ptr_matrix->data(x, y) = c1 * cos((cimg::PI / 2 / N) * y * (2 * x + 1));
        }
    }
    return ptr_matrix;
}

unsigned long long ph_dct_imagehash(CImg<uint8_t> src) {
    CImg<float> meanfilter(7, 7, 1, 1, 1);
    CImg<float> img;
    if (src.spectrum() == 3) {
        img = src.get_RGBtoYCbCr()
            .channel(0)
            .get_convolve(meanfilter);
    } else if (src.spectrum() == 4) {
        int width = src.width();
        int height = src.height();
        int depth = src.depth();
        img = src.get_crop(0, 0, 0, 0, width - 1, height - 1, depth - 1, 2)
                  .RGBtoYCbCr()
                  .channel(0)
                  .get_convolve(meanfilter);
    } else {
        img = src.get_channel(0).get_convolve(meanfilter);
    }

    img.resize(32, 32);
    CImg<float> *C = ph_dct_matrix(32);
    CImg<float> Ctransp = C->get_transpose();
    CImg<float> dctImage = (*C) * img * Ctransp;
    CImg<float> subsec = dctImage.crop(1, 1, 8, 8).unroll('x');

    float median = subsec.median();
    unsigned long long one = 0x0000000000000001;
    unsigned long long hash = 0x0000000000000000;
    for (int i = 0; i < 64; i++) {
        float current = subsec(i);
        if (current > median) hash |= one;
        one = one << 1;
    }

    delete C;
    return hash;
}

#endif /// PyImgC_IMP_PHASH_H