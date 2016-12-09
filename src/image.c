#include <stdlib.h>

#include <png.h>

#include "../include/image.h"

#define SIG_RNUM 8
#define COLOR_DEPTH 8

void plpo_image_init(PLPOImage_t* img) {
    img->width      = -1;
    img->height     = -1;
    img->row_length = -1;
    img->color_type = -1;
    img->bytes      = NULL;
}

void plpo_image_destroy(PLPOImage_t* img) {
    free(img->bytes);
    free(img);
    img = NULL;
}

int plpo_image_read(const char* path, PLPOImage_t* img) {
    FILE* f = fopen(path, "rb");
    char* header = (char*) malloc(sizeof(char) * SIG_RNUM);
    int n = fread(header, 1, SIG_RNUM, f);
    if (n != SIG_RNUM) {
        free(header);
        fclose(f);
        return -1;
    }

    int h_sig = png_sig_cmp((png_bytep)header, 0, SIG_RNUM);
    if (h_sig != 0) {
        free(header);
        fclose(f);
        return -1;
    }
    free(header);
    header = NULL;

    png_structp img_p = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!img_p) {
        fclose(f);
        return -1;
    }

    png_infop info = png_create_info_struct(img_p);
    if (!info) {
        png_destroy_read_struct(&img_p, (png_infopp)NULL, (png_infopp)NULL);
        fclose(f);
        return -1;
    }

    if (setjmp(png_jmpbuf(img_p))) {
        png_destroy_read_struct(&img_p, (png_infopp)NULL, (png_infopp)NULL);
        fclose(f);
        return -1;
    }

    png_init_io(img_p, f);
    png_set_sig_bytes(img_p, SIG_RNUM);
    png_read_info(img_p, info);

    png_byte b_depth = png_get_bit_depth(img_p, info);
    png_byte color_type = png_get_color_type(img_p, info);
    if (color_type == PNG_COLOR_TYPE_PALETTE) png_set_palette_to_rgb(img_p);
    if (color_type == PNG_COLOR_TYPE_GRAY && b_depth < 8) png_set_expand_gray_1_2_4_to_8(img_p);
    if (color_type == PNG_COLOR_TYPE_GRAY ||
            color_type == PNG_COLOR_TYPE_GRAY_ALPHA) png_set_gray_to_rgb(img_p);
    if (color_type == PNG_COLOR_TYPE_RGB ||
            color_type == PNG_COLOR_TYPE_GRAY ||
            color_type == PNG_COLOR_TYPE_PALETTE) png_set_gray_to_rgb(img_p);
    if (color_type == PNG_COLOR_TYPE_GRAY ||
            color_type == PNG_COLOR_TYPE_GRAY_ALPHA) png_set_gray_to_rgb(img_p);

    png_read_update_info(img_p, info);
    img->color_type = color_type;
    img->width  = png_get_image_width(img_p, info);
    img->height = png_get_image_height(img_p, info);
    img->row_length = (int) png_get_rowbytes(img_p, info);
    img->bytes = calloc(img->height, sizeof(png_bytep));
    for (int y = 0; y < img->height; ++y) {
        img->bytes[y] = malloc(sizeof(png_byte) * img->row_length);
    }
    png_read_image(img_p, img->bytes);
    png_read_end(img_p, info);

    fclose(f);
    png_destroy_read_struct(&img_p, &info, NULL);

    return 0;
}

int plpo_image_write(const char* path, PLPOImage_t* img) {
    if (!img->bytes) {
        return -1;
    }

    FILE* f = fopen(path, "wb");
    if (!f) {
        return -1;
    }

    png_structp img_o = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!img_o) {
        fclose(f);
        return -1;
    }

    png_infop info = png_create_info_struct(img_o);
    if (!info) {
        fclose(f);
        return -1;
    }

    if (setjmp(png_jmpbuf(img_o))) {
        png_destroy_read_struct(&img_o, &info, NULL);
        fclose(f);
        return -1;
    }

    png_init_io(img_o, f);
    png_set_IHDR(
            img_o,
            info,
            img->width,
            img->height,
            COLOR_DEPTH,
            img->color_type,
            PNG_INTERLACE_ADAM7,
            PNG_COMPRESSION_TYPE_DEFAULT,
            PNG_FILTER_TYPE_DEFAULT
            );
    png_write_info(img_o, info);
    png_write_image(img_o, img->bytes);
    
    fclose(f);

    return 0;
}
