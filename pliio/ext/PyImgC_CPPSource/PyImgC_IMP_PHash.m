
#include "PyImgC_IMP_PHash.h"
using namespace std;

int ph_radon_projections(const CImg<uint8_t> &img, int N, Projections &projs) {

    int width = img.width();
    int height = img.height();
    int D = (width > height) ? width : height;
    float x_center = static_cast<float>(width / 2);
    float y_center = static_cast<float>(height / 2);
    int x_off = static_cast<int>(std::floor(x_center + ROUNDING_FACTOR(x_center)));
    int y_off = static_cast<int>(std::floor(y_center + ROUNDING_FACTOR(y_center)));

    projs.R = new CImg<uint8_t>(N, D, 1, 1, 0);
    projs.nb_pix_perline = (int *)PyMem_Calloc(N, sizeof(int));

    if (!projs.R || !projs.nb_pix_perline) { return EXIT_FAILURE; }

    projs.size = N;

    CImg<uint8_t> *ptr_radon_map = projs.R;
    int *nb_per_line = projs.nb_pix_perline;

    for (int k = 0; k < N / 4 + 1; k++) {
        double theta = k * cimg::PI / N;
        double alpha = std::tan(theta);
        for (int x = 0; x < D; x++) {
            double y = alpha * (x - x_off);
            int yd = static_cast<int>(std::floor(y + ROUNDING_FACTOR(y)));
            if ((yd + y_off >= 0) && (yd + y_off < height) && (x < width)) {
                *ptr_radon_map->data(k, x) = img(x, yd + y_off);
                nb_per_line[k] += 1;
            }
            if ((yd + x_off >= 0) && (yd + x_off < width) && (k != N / 4)
                && (x < height)) {
                *ptr_radon_map->data(N / 2 - k, x) = img(yd + x_off, x);
                nb_per_line[N / 2 - k] += 1;
            }
        }
    }
    
    int j = 0;
    for (int k = 3 * N / 4; k < N; k++) {
        double theta = k * cimg::PI / N;
        double alpha = std::tan(theta);
        for (int x = 0; x < D; x++) {
            double y = alpha * (x - x_off);
            int yd = static_cast<int>(std::floor(y + ROUNDING_FACTOR(y)));
            if ((yd + y_off >= 0) && (yd + y_off < height) && (x < width)) {
                *ptr_radon_map->data(k, x) = img(x, yd + y_off);
                nb_per_line[k] += 1;
            }
            if ((y_off - yd >= 0) && (y_off - yd < width)
                && (2 * y_off - x >= 0) && (2 * y_off - x < height)
                && (k != 3 * N / 4)) {
                *ptr_radon_map->data(k - j, x)
                    = img(-yd + y_off, -(x - y_off) + y_off);
                nb_per_line[k - j] += 1;
            }
        }
        j += 2;
    }

    return EXIT_SUCCESS;
}

int ph_feature_vector(const Projections &projs, Features &fv) {

    CImg<uint8_t> *ptr_map = projs.R;
    CImg<uint8_t> projection_map = *ptr_map;
    int *nb_perline = projs.nb_pix_perline;
    int N = projs.size;
    int D = projection_map.height();

    fv.features = (double *)PyMem_Malloc(N * sizeof(double));
    fv.size = N;
    if (!fv.features) { return EXIT_FAILURE; }

    double *feat_v = fv.features;
    double sum = 0.0;
    double sum_sqd = 0.0;
    for (int k = 0; k < N; k++) {
        double line_sum = 0.0;
        double line_sum_sqd = 0.0;
        int nb_pixels = nb_perline[k];
        for (int i = 0; i < D; i++) {
            line_sum += projection_map(k, i);
            line_sum_sqd += projection_map(k, i) * projection_map(k, i);
        }
        feat_v[k] = (line_sum_sqd / nb_pixels) - (line_sum * line_sum)
                                                 / (nb_pixels * nb_pixels);
        sum += feat_v[k];
        sum_sqd += feat_v[k] * feat_v[k];
    }
    double mean = sum / N;
    double var = sqrt((sum_sqd / N) - (sum * sum) / (N * N));

    for (int i = 0; i < N; i++) {
        feat_v[i] = (feat_v[i] - mean) / var;
    }

    return EXIT_SUCCESS;
}

int ph_dct(const Features &fv, Digest &digest) {

    int N = fv.size;
    const int nb_coeffs = 40;

    digest.coeffs = (uint8_t *)PyMem_Malloc(nb_coeffs * sizeof(uint8_t));
    if (!digest.coeffs) { return EXIT_FAILURE; }

    digest.size = nb_coeffs;
    double *R = fv.features;
    uint8_t *D = digest.coeffs;

    double D_temp[nb_coeffs];
    double max = 0.0;
    double min = 0.0;
    for (int k = 0; k < nb_coeffs; k++) {
        double sum = 0.0;
        for (int n = 0; n < N; n++) {
            double temp = R[n] * cos((cimg::PI * (2 * n + 1) * k) / (2 * N));
            sum += temp;
        }
        if (k == 0) {
            D_temp[k] = sum / sqrt(static_cast<double>(N));
        } else {
            D_temp[k] = sum * SQRT_TWO / sqrt(static_cast<double>(N));
        }
        if (D_temp[k] > max) { max = D_temp[k]; }
        if (D_temp[k] < min) { min = D_temp[k]; }
    }

    for (int i = 0; i < nb_coeffs; i++) {
        D[i] = static_cast<uint8_t>(UCHAR_MAX * (D_temp[i] - min) / (max - min));
    }

    return EXIT_SUCCESS;
}

int ph_crosscorr(const Digest &x, const Digest &y, double &pcc, double threshold) {

    int N = y.size;
    int result = 0;

    uint8_t *x_coeffs = x.coeffs;
    uint8_t *y_coeffs = y.coeffs;

    double *r = new double[N];
    double sumx = 0.0;
    double sumy = 0.0;
    
    for (int i = 0; i < N; i++) {
        sumx += x_coeffs[i];
        sumy += y_coeffs[i];
    }
    
    double meanx = sumx / N;
    double meany = sumy / N;
    double max = 0;
    for (int d = 0; d < N; d++) {
        double num = 0.0;
        double denx = 0.0;
        double deny = 0.0;
        for (int i = 0; i < N; i++) {
            num += (x_coeffs[i] - meanx) * (y_coeffs[(N + i - d) % N] - meany);
            denx += pow((x_coeffs[i] - meanx), 2);
            deny += pow((y_coeffs[(N + i - d) % N] - meany), 2);
        }
        r[d] = num / sqrt(denx * deny);
        if (r[d] > max) max = r[d];
    }
    
    delete[] r;
    pcc = max;
    if (max > threshold) { result = 1; }

    return result;
}

int ph_image_digest(const CImg<uint8_t> &img, double sigma, double gamma,
                     Digest &digest, int N) {

    int result = EXIT_FAILURE;
    CImg<uint8_t> graysc;

    if (img.spectrum() >= 3) {
        graysc = img.get_RGBtoYCbCr().channel(0);
    } else if (img.spectrum() == 1) {
        graysc = img;
    } else {
        return result;
    }

    graysc.blur((float)sigma);

    (graysc / graysc.max()).pow(gamma);

    Projections projs;
    if (ph_radon_projections(graysc, N, projs) < 0) { goto cleanup; }

    Features features;
    if (ph_feature_vector(projs, features) < 0) { goto cleanup; }
    if (ph_dct(features, digest) < 0) { goto cleanup; }
    result = EXIT_SUCCESS;

cleanup:
    PyMem_Free(projs.nb_pix_perline);
    PyMem_Free(features.features);

    delete projs.R;
    return result;
}

int ph_compare_images(const CImg<uint8_t> &imA, const CImg<uint8_t> &imB,
                       double &pcc,
                       double sigma, double gamma,
                       int N, double threshold) {

    int result = 0;
    Digest digestA;
    if (ph_image_digest(imA, sigma, gamma, digestA, N) < 0) { goto cleanup; }

    Digest digestB;
    if (ph_image_digest(imB, sigma, gamma, digestB, N) < 0) { goto cleanup; }
    if (ph_crosscorr(digestA, digestB, pcc, threshold) < 0) { goto cleanup; }
    if (pcc > threshold) { result = 1; }

cleanup:
    PyMem_Free(digestA.coeffs);
    PyMem_Free(digestB.coeffs);
    return result;
}

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

    static const CImg<float> meanfilter(7, 7, 1, 1, 1);
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

int ph_hamming_distance(const unsigned long long hash1, const unsigned long long hash2) {

    unsigned long long x = hash1 ^ hash2;
    const unsigned long long m1 = 0x5555555555555555ULL;
    const unsigned long long m2 = 0x3333333333333333ULL;
    const unsigned long long h01 = 0x0101010101010101ULL;
    const unsigned long long m4 = 0x0f0f0f0f0f0f0f0fULL;
    x -= (x >> 1) & m1;
    x = (x & m2) + ((x >> 2) & m2);
    x = (x + (x >> 4)) & m4;
    return (x * h01) >> 56;
}

CImg<float> *ph_mh_kernel(float alpha, float level) {

    int sigma = static_cast<int>(4 * pow((float)alpha, (float)level));
    static CImg<float> *pkernel = NULL;
    float xpos, ypos, A;
    if (!pkernel) {
        pkernel = new CImg<float>(2 * sigma + 1, 2 * sigma + 1, 1, 1, 0);
        cimg_forXY(*pkernel, X, Y) {
            xpos = pow(alpha, -level) * (X - sigma);
            ypos = pow(alpha, -level) * (Y - sigma);
            A = xpos * xpos + ypos * ypos;
            pkernel->atXY(X, Y) = (2 - A) * exp(-A / 2);
        }
    }
    return pkernel;
}

uint8_t *ph_mh_imagehash(CImg<uint8_t> src, float alpha, float lvl) {

    uint8_t *hash = (unsigned char *)PyMem_Malloc(72 * sizeof(uint8_t));
    CImg<uint8_t> img;
    
    if (src.spectrum() == 3) {
        img = src.get_RGBtoYCbCr()
                  .channel(0)
                  .blur(1.0)
                  .resize(512, 512, 1, 1, 5)
                  .get_equalize(256);
    } else {
        img = src.get_channel(0)
                  .get_blur(1.0)
                  .resize(512, 512, 1, 1, 5)
                  .get_equalize(256);
    }
    
    CImg<float> *pkernel = ph_mh_kernel(alpha, lvl);
    CImg<float> fresp = img.get_correlate(*pkernel);
    img.clear();
    fresp.normalize(0, 1.0);
    CImg<float> blocks(31, 31, 1, 1, 0);
    for (int rindex = 0; rindex < 31; rindex++) {
        for (int cindex = 0; cindex < 31; cindex++) {
            blocks(rindex, cindex)
                = fresp.get_crop(rindex * 16, cindex * 16, rindex * 16 + 16 - 1,
                                 cindex * 16 + 16 - 1).sum();
        }
    }
    int hash_index;
    int nb_ones = 0, nb_zeros = 0;
    int bit_index = 0;
    unsigned char hashbyte = 0;
    for (int rindex = 0; rindex < 31 - 2; rindex += 4) {
        CImg<float> subsec;
        for (int cindex = 0; cindex < 31 - 2; cindex += 4) {
            subsec = blocks.get_crop(cindex, rindex, cindex + 2, rindex + 2)
                         .unroll('x');
            float ave = subsec.mean();
            cimg_forX(subsec, I) {
                hashbyte <<= 1;
                if (subsec(I) > ave) {
                    hashbyte |= 0x01;
                    nb_ones++;
                } else {
                    nb_zeros++;
                }
                bit_index++;
                if ((bit_index % 8) == 0) {
                    hash_index = static_cast<int>((bit_index / 8) - 1);
                    hash[hash_index] = hashbyte;
                    hashbyte = 0x00;
                }
            }
        }
    }
    return hash;
}

int ph_bitcount8(uint8_t val) {

    int num = 0;
    while (val) {
        ++num;
        val &= val - 1;
    }
    return num;
}

double ph_hamming_distance2(uint8_t *hashA, uint8_t *hashB, int lenA, int lenB) {

    if (lenA != lenB) { return -1.0; }
    if ((hashA == NULL) || (hashB == NULL) || (lenA <= 0)) { return -1.0; }
    double dist = 0;
    uint8_t D = 0;
    for (int i = 0; i < lenA; i++) {
        D = hashA[i] ^ hashB[i];
        dist = dist + static_cast<double>(ph_bitcount8(D));
    }
    double bits = static_cast<double>(lenA) * 8;
    return dist / bits;
}
