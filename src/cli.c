#include <stdio.h>
#include <stdlib.h>

#include "../include/cli.h"
#include "../include/image.h"
#include "../include/plpo.h"

int read_args(int argvc, char** argv, PLPO_CLIArgs_t* args) {
    if (argvc < 4) {
        return -1;
    }

    args->image_path = argv[1];
    args->filter_path = argv[2];
    args->out_image_path = argv[3];

    return 0;
}

int start_cli(int argvc, char** argv) {
    int err = 0;

    if (argvc < 2) {
        return -1;
    }
    
    PLPO_CLIArgs_t args;
    if ((err = read_args(argvc, argv, &args)) != 0) {
        return err;
    }

    PLPOImage_t* img = malloc(sizeof(*img));
    plpo_image_init(img);
    if ((err = plpo_image_read(args.image_path, img)) != 0) {
        return err;
    }

    PLPOImage_t* lut = malloc(sizeof(*lut));
    plpo_image_init(lut);
    if ((err = plpo_image_read(args.filter_path, lut)) != 0) {
        return err;
    }
        
    if ((err = plpo_make_trilerp(lut, img)) != 0) {
        plpo_image_destroy(img);
        plpo_image_destroy(lut);
        return err;
    }
    
    if ((err = plpo_image_write(args.out_image_path, img)) != 0) {
        plpo_image_destroy(img);
        plpo_image_destroy(lut);
        return err;
    }
    
    plpo_image_destroy(img);
    plpo_image_destroy(lut);

    fprintf(stderr, "Image has been saved to: %s\n", args.out_image_path);

    return 0;
}
