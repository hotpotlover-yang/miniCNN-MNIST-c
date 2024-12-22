#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <src/dataloder.h>


#define TRAIN_IMG_PATH "data/train-images.idx3-ubyte"
#define TRAIN_LBL_PATH "data/train-labels.idx1-ubyte"

#define INPUT_SIZE 784




int main(int argc, char const *argv[])
{

    srand(time(NULL));

    DataLoader dataloader;
    dataloader_init(&dataloader, TRAIN_IMG_PATH, TRAIN_LBL_PATH, 1);



    free(dataloader.images);
    free(dataloader.labels);
    return 0;
}
