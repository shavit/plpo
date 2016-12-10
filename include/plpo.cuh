#ifndef PLPO_CUH
#define PLPO_CUH

__global__
void plpo_make_trilerp(unsigned char* lut_mat, unsigned char* img_mat, int w, int h, const int m, const int m2, float sig, int ch);

#endif // !PLPO_CUH
