#include <stdio.h>

#include "../include/cli.h"

int main(int argvc, char** argv) {
    if (start_cli(argvc, argv) != 0) {
        fprintf(stderr, "Could not start CLI\n");
        return -1;
    }

    return 0;
}
