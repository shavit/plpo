#ifndef PLPO_INTERPOLATE_CUH
#define PLPO_INTERPOLATE_CUH

typedef struct Pixel {
    //int r;
    //int g;
    //int b;
    float r;
    float g;
    float b;
} Pixel_t;

typedef struct Lattice3D {
    int x0;
    int x1;
    int y0;
    int y1;
    int z0;
    int z1;
    Pixel_t c000;
    Pixel_t c001;
    Pixel_t c010;
    Pixel_t c011;
    Pixel_t c100;
    Pixel_t c101;
    Pixel_t c110;
    Pixel_t c111;
} Lattice3D_t;

__device__
Pixel_t get_pp(unsigned char* bytes[], size_t pitch, const int* m, const int* m2, const int* x, const int* y, const int* z);
__device__
Lattice3D_t create_lattice3d(float* bytes, size_t pitch, const int* m, const int* m2, const float* x, const float* y, const float* z);
__device__
Pixel_t interpolate_lattice3d(Lattice3D_t* lat, const float* r, const float* g, const float* b);
__device__
float trilrp(float c000, float c001, float c010, float c011, float c100, float c101, float c110, float c111, float xd, float yd, float zd);

#endif // !PLPO_INTERPOLATE_CUH

