#include <math.h>

#include "../include/plpo.h"
#include "../include/interpolate.h"

int plpo_make_trilerp(PLPOImage_t* lut, PLPOImage_t* img) {
    const int m = floor(cbrt(lut->width));
    const int m2 = m * m; // block range 0..m2
    const float sig = m2 - 1;
    
    float r, g, b;
    for (int i = 0; i < img->height; ++i) {
        for (int j = 0; j < img->row_length; ++j) {
            r = img->bytes[i][3 * j] * sig;
            g = img->bytes[i][3 * j + 1] * sig;
            b = img->bytes[i][3 * j + 2] * sig;
            Lattice3D_t lat = create_lattice3d(img->bytes, &m, &m2, &r, &g, &b);
            Pixel_t px = interpolate_lattice3d(&lat, img->bytes, r, g, b);
        }
    }

    return 0;
}
