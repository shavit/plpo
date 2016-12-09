#ifndef PLPO_INTERPOLATE_H
#define PLPO_INTERPOLATE_H

typedef struct Pixel {
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

Pixel_t get_pp(unsigned char* bytes[], const int* m, const int* m2, const int* x, const int* y, const int* z);
Lattice3D_t create_lattice3d(unsigned char* bytes[], const int* m, const int* m2, float* x, float* y, float* z);
Pixel_t interpolate_lattice3d(Lattice3D_t* lat, float* r, float* g, float* b);
float trilrp(float c000, float c001, float c010, float c011, float c100, float c101, float c110, float c111, float xd, float yd, float zd);

#endif // !PLPO_INTERPOLATE_H

