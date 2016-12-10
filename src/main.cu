#include <cstdlib>
#include <stdio.h>
#include <math.h>

#include "../include/interpolate.cuh"
#include "../include/plpo.cuh"

extern "C" {
#include "../include/cli.h"
#include "../include/image.h"
}

__host__
int read_args(int argvc, char** argv, PLPO_CLIArgs_t* args) {
    if (argvc < 4) {
        return -1;
    }

    args->image_path = argv[1];
    args->filter_path = argv[2];
    args->out_image_path = argv[3];
    args->parallel = false;

    return 0;
}

void mem_assign_unflatten(unsigned char** output, unsigned char* input, const int w, const int h) {
    for (int i = 0; i < h; ++i) {
        for (int j = 0; j < w; ++j) {
            const int xy = i * w + j;
            output[i][j] = input[xy];
        }
    }
}

void mem_assign_flatten(unsigned char* output, unsigned char** input, const int w, const int h) {
    for (int i = 0; i < h; ++i) {
        for (int j = 0; j < w; ++j) {
            const int xy = i * w + j;
            output[xy] = input[i][j];
        }
    }
}

void checkCudaError(const char* msg) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Error: %s\n", msg);
        fprintf(stderr, "CUDA Error %d: %s\n", err, cudaGetErrorString(err));

        exit(5);
    }
}

void determineCudaMaxBlockSize(int* numBlocks, int* maxBlockSize) {
    cudaOccupancyMaxPotentialBlockSize(numBlocks, maxBlockSize, plpo_make_trilerp);
    //fprintf(stderr, "Determined cuda potential block size.\n\t%d blocks (numBlocks)\n\t%d grid size (maxBlockSize)\n", *numBlocks, *maxBlockSize);
}

__host__
int main(int argvc, char** argv) {
    int err = 0;

    PLPO_CLIArgs_t* args = (PLPO_CLIArgs_t*) malloc(sizeof(*args));
    if ((err = read_args(argvc, argv, args)) != 0) {
        return err;
    }

    PLPOImage_t* img = (PLPOImage_t*) malloc(sizeof(*img));
    plpo_image_init(img);
    if ((err = plpo_image_read(args->image_path, img)) != 0) {
        return err;
    }
    unsigned char* img_1d = (unsigned char*) malloc(sizeof(unsigned char) * img->row_length * img->height);
    mem_assign_flatten(img_1d, img->bytes, img->row_length, img->height);

    PLPOImage_t* lut = (PLPOImage_t*) malloc(sizeof(*lut));
    plpo_image_init(lut);
    if ((err = plpo_image_read(args->filter_path, lut)) != 0) {
        return err;
    }
    unsigned char* lut_1d = (unsigned char*) malloc(sizeof(unsigned char) * lut->row_length * lut->height);
    mem_assign_flatten(lut_1d, lut->bytes, lut->row_length, lut->height);

    int numBlocks;
    int maxBlockSize;
    determineCudaMaxBlockSize(&numBlocks, &maxBlockSize);
    const dim3 dim_block(maxBlockSize / 32 / 2, maxBlockSize / 32 / 2, 1);
    dim3 dim_grid((img->height - 1) / dim_block.x + 1, (img->width - 1) / dim_block.y + 1, 1); // round up

    unsigned char* cuimg_mat;
    cudaMalloc((void**) &cuimg_mat, sizeof(unsigned char) * img->row_length * img->height);
    checkCudaError("After allocating img");
    cudaMemcpy(cuimg_mat, img_1d, sizeof(unsigned char) * img->row_length * img->height, cudaMemcpyHostToDevice);
    checkCudaError("After copying img row to device");

    unsigned char* culut_mat;
    cudaMalloc((void**) &culut_mat, sizeof(unsigned char) * lut->row_length * lut->height);
    checkCudaError("After allocating lut");
    cudaMemcpy(culut_mat, lut_1d, sizeof(unsigned char) * lut->row_length * lut->height, cudaMemcpyHostToDevice);
    checkCudaError("After copying lut row to device");

    const int m = floor(cbrt(lut->width));
    const int m2 = m * m; // block range 0..m2
    const float sig = (m2 - 1) / (float)0xff;
    plpo_make_trilerp<<<dim_grid, dim_block>>>(culut_mat, cuimg_mat, img->width, img->height, m, m2, sig, 3);
    checkCudaError("Error returning from kernel");

    cudaMemcpy(img_1d, cuimg_mat, sizeof(unsigned char) * img->row_length * img->height, cudaMemcpyDeviceToHost);
    checkCudaError("Error after data copied to host");
    mem_assign_unflatten(img->bytes, img_1d, img->row_length, img->height);

    cudaFree(cuimg_mat);
    cudaFree(culut_mat);
    free(img_1d);
    free(lut_1d);

    if ((err = plpo_image_write(args->out_image_path, img)) != 0) {
        plpo_image_destroy(img);
        plpo_image_destroy(lut);
        return err;
    }

    plpo_image_destroy(img);
    plpo_image_destroy(lut);

    fprintf(stderr, "Saved %s\n", args->out_image_path);

    return 0;
}    
