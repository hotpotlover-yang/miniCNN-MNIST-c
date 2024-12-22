#ifndef UTILS_H
#define UTILS_H
#include<stdio.h>
#include <stdlib.h>


inline FILE* fopen_file(const char* path, const char* mode, const char* file, int line) {
    FILE* fp = fopen(path, mode);
    if (file == NULL) {
        printf("Error opening file %s\n", path);
        fprintf(stderr, "Error: Failed to open file '%s' at %s:%d\n", path, file, line);
        fprintf(stderr, "Error details:\n");
        fprintf(stderr, "  File: %s\n", file);
        fprintf(stderr, "  Line: %d\n", line);
        fprintf(stderr, "  Path: %s\n", path);
        fprintf(stderr, "  Mode: %s\n", mode);
        fprintf(stderr, "---> HINT 1: dataset files/code have moved to dev/data recently (May 20, 2024). You may have to mv them from the legacy data/ dir to dev/data/(dataset), or re-run the data preprocessing script. Refer back to the main README\n");
        fprintf(stderr, "---> HINT 2: possibly try to re-run `python train_gpt2.py`\n");
        exit(EXIT_FAILURE);
        exit(1);
    }
    return fp;
}
#define fopenCheck(path, mode) fopen_file(path, mode, __FILE__, __LINE__)

inline void* fread_check(void *ptr, size_t size, size_t nmemb, FILE *stream, const char* file, int line) {
    size_t ret = fread(ptr, size, nmemb, stream);
    if (ret != nmemb) {
        fprintf(stderr, "Error: Failed to read file '%s' at %s:%d\n", file, line);
        fprintf(stderr, "Error details:\n");
        fprintf(stderr, "  File: %s\n", file);
        fprintf(stderr, "  Line: %d\n", line);
        fprintf(stderr, "  Read: %lu\n", ret);
        fprintf(stderr, "  Expected: %lu\n", nmemb);
        exit(EXIT_FAILURE);
    }
    return ptr;
}
#define freadCheck(ptr, size, nmemb, stream) fread_check(ptr, size, nmemb, stream, __FILE__, __LINE__)

inline void fclose_check(FILE *fp, const char* file, int line) {
    if (fclose(fp) != 0) {
        fprintf(stderr, "Error: Failed to close file '%s' at %s:%d\n", file, line);
        fprintf(stderr, "Error details:\n");
        fprintf(stderr, "  File: %s\n", file);
        fprintf(stderr, "  Line: %d\n", line);
        exit(EXIT_FAILURE);
    }
}
#define fcloseCheck(fp) fclose_check(fp, __FILE__, __LINE__)
#endif