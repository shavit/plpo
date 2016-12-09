#ifndef PLPO_IMAGE_H
#define PLPO_IMAGE_H

typedef struct PLPOImage {
    int width;
    int height;
    int row_length;
    int color_type;
    unsigned char** bytes;
} PLPOImage_t;

void plpo_image_init(PLPOImage_t* img);
void plpo_image_destroy(PLPOImage_t* img);
int plpo_image_read(const char* path, PLPOImage_t* img);
int plpo_image_write(const char* path, PLPOImage_t* img);

#endif // !PLPO_IMAGE_H

