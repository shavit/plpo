#include <math.h>

#include "../include/interpolate.h"

int min(int a, int b) {
    if (a > b) {
        return b;
    } else {
        return a;
    }
}

int max(int a, int b) {
    if (a > b) {
        return a;
    } else {
        return b;
    }
}

Pixel_t get_pp(unsigned char* bytes[], const int* m, const int* m2, const int* x, const int* y, const int* z) {
    const int u = (*z % *m) * *m2;
    const int v = (*z / *m) * *m2;
    const int ui = (u + *x) * 3;
    const int vi = (v + *y);
    
    Pixel_t p;
    p.r = bytes[vi][ui];
    p.g = bytes[vi][ui + 1];
    p.b = bytes[vi][ui + 2];
    
    return p;
}

Lattice3D_t create_lattice3d(unsigned char* bytes[], const int* m, const int* m2, float* x, float* y, float* z) {
    Lattice3D_t lat;
    
    lat.x0 = max(0, floor(*x - 1));
    lat.x1 = min(*m2 - 1, floor(*x + 1));
    lat.y0 = max(0, floor(*y - 1));
    lat.y1 = min(*m2 - 1, floor(*y + 1));
    lat.z0 = max(0, floor(*z - 1));
    lat.z1 = min(*m2 - 1, floor(*z + 1));

    lat.c000 = get_pp(bytes, m, m2, &lat.x0, &lat.y0, &lat.z0);
    lat.c001 = get_pp(bytes, m, m2, &lat.x0, &lat.y0, &lat.z1);
    lat.c010 = get_pp(bytes, m, m2, &lat.x0, &lat.y1, &lat.z0);
    lat.c011 = get_pp(bytes, m, m2, &lat.x0, &lat.y1, &lat.z1);
    lat.c100 = get_pp(bytes, m, m2, &lat.x1, &lat.y0, &lat.z0);
    lat.c101 = get_pp(bytes, m, m2, &lat.x1, &lat.y0, &lat.z1);
    lat.c110 = get_pp(bytes, m, m2, &lat.x1, &lat.y1, &lat.z0);
    lat.c111 = get_pp(bytes, m, m2, &lat.x1, &lat.y1, &lat.z1);
    
    return lat;
}

float trilrp(float c000, float c001, float c010, float c011, float c100, float c101, float c110, float c111, float xd, float yd, float zd) {
    const float c00 = c000 * (1 - xd) + c100 * xd;
    const float c01 = c001 * (1 - xd) + c101 * xd;
    const float c10 = c010 * (1 - xd) + c111 * xd;
    const float c11 = c011 * (1 - xd) + c111 * xd;
    const float c0 = c00 * (1 - yd) + c10 * yd;
    const float c1 = c01 * (1 - yd) + c11 * yd;
    const float c = c0 * (1 - zd) + c1 * zd;

    return c;
}

Pixel_t interpolate_lattice3d(Lattice3D_t* lat, float* r, float* g, float* b) {
    const float xd = (*r - lat->x0) / (lat->x1 - lat->x0);
    const float yd = (*g - lat->y0) / (lat->y1 - lat->y0);
    const float zd = (*b - lat->z0) / (lat->z1 - lat->z0);

    Pixel_t p;
    p.r = trilrp(
            lat->c000.r, 
            lat->c001.r,
            lat->c010.r,
            lat->c011.r,
            lat->c100.r,
            lat->c101.r,
            lat->c110.r,
            lat->c111.r,
            xd, yd, zd);
    p.g = trilrp(
            lat->c000.g, 
            lat->c001.g,
            lat->c010.g,
            lat->c011.g,
            lat->c100.g,
            lat->c101.g,
            lat->c110.g,
            lat->c111.g,
            xd, yd, zd);
    p.b = trilrp(
            lat->c000.b, 
            lat->c001.b,
            lat->c010.b,
            lat->c011.b,
            lat->c100.b,
            lat->c101.b,
            lat->c110.b,
            lat->c111.b,
            xd, yd, zd);

    return p;
}
