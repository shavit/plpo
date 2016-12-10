#ifndef PLPO_CLI_H
#define PLPO_CLI_H

#include <stdbool.h>

typedef struct PLPO_CLIArgs {
    char* image_path;
    char* filter_path;
    char* out_image_path;
    bool parallel;
} PLPO_CLIArgs_t;

int start_cli(int argvc, char** argv);

#endif // !PLPO_CLI_H
