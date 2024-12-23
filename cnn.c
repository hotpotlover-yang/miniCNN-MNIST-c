#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <dataloader.h>


#define TRAIN_IMG_PATH "data/train-images.idx3-ubyte"
#define TRAIN_LBL_PATH "data/train-labels.idx1-ubyte"

#define INPUT_SIZE 784

#define ImageSize 28
#define K1 5
#define C1 16
#define P1 2
#define k2 5
#define C2 36
#define P2 2
#define FC1_SIZE 128
#define OUTPUT_SIZE 10

#define EPOCHS 20
#define BATCH 5
#define TRAIN_SPLIT 0.8
#define LEARN_RATE 0.05

typedef struct{
    float* weights;
    int size;
    int stride;
    int filters;
} Conv;

typedef struct{
    float* weights;
    int input_size;
    int output_size;
}FC;

typedef struct{
    Conv conv1; // (K1,K1,C1)
    int pool1Size;
    Conv Conv2; // (K2,K2,C2)
    int pool2Size;
    // Conv Conv3; // (K3,K3,C3)
    FC fc;
    FC output;

}Paramerters;

typedef struct{
    float* conv1; // ( (imageSize-K1)/stride + 1, (imageSize-K1)/stride + 1, C1)
    float* relu1; // ( (imageSize-K1)/stride + 1, (imageSize-K1)/stride + 1, C1)
    float* pool1; // ( ((imageSize-K1)/stride + 1)/P1, ((imageSize-K1)/stride + 1)/P1, C1)
    float* conv2; // ( ((imageSize-K1)/stride + 1)/P1 - K2)/stride + 1, ((imageSize-K1)/stride + 1)/P1 - K2)/stride + 1, C2)
    float* relu2; // ( ((imageSize-K1)/stride + 1)/P1 - K2)/stride + 1, ((imageSize-K1)/stride + 1)/P1 - K2)/stride + 1, C2)
    float* pool2; // ( (((imageSize-K1)/stride + 1)/P1 - K2)/stride + 1)/P2, (((imageSize-K1)/stride + 1)/P1 - K2)/stride + 1)/P2, C2)
    float* fc; // (10,1,1)
}Activation;

typedef struct
{
    float* data; // (imageSize, imageSize)
    int imageSize;
    Paramerters params;
    float* params_memory;
    int toal_params;
    Activation acts;
}CNN;

void init_params(float*params, int size){
    float scale = sqrt(2.0f/size);
    for(int i = 0; i < size; i++){
        params[i] = 2*scale * ((float)rand()/(float)(RAND_MAX) - 0.5f);
    }
}

void CNN_init(CNN *model, int imageSize,int k1, int c1, int stride1, int p1,int K2,int c2, int stride2,int p2,int fc1_size, int outputSize, int batch){
    printf("init model\n");
    model->imageSize = imageSize;
    model->params.conv1.size = K1;
    model->params.conv1.stride = stride1;
    model->params.conv1.filters = C1;
    model->params.pool1Size = P1;
    model->params.Conv2.size = K2;
    model->params.Conv2.stride = stride2;
    model->params.Conv2.filters = C2;
    model->params.pool2Size = P2;
    model->params.fc.input_size = ((((imageSize - K1)/stride1 + 1)/P1 - K2)/stride2 + 1)*((((imageSize - K1)/stride1 + 1)/P1 - K2)/stride2 + 1)*C2;
    model->params.fc.output_size = fc1_size;
    model->params.output.input_size = fc1_size;
    model->params.output.output_size = outputSize;

    int total_params = K1*K1*C1 + K2*K2*C2 + model->params.fc.input_size*fc1_size + fc1_size*outputSize;
    model->toal_params = total_params;
    printf("total params: %d\n", total_params);
    model->params_memory = (float*)malloc(total_params * sizeof(float));
    // init weights
    model->params.conv1.weights = model->params_memory;
    init_params(model->params.conv1.weights, K1*K1*C1);
    model->params.Conv2.weights = model->params_memory + K1*K1*C1;
    init_params(model->params.Conv2.weights, K2*K2*C2);
    model->params.fc.weights = model->params_memory + K1*K1*C1 + K2*K2*C2;
    init_params(model->params.fc.weights, model->params.fc.input_size*model->params.fc.output_size);
    model->params.output.weights = model->params_memory + K1*K1*C1 + K2*K2*C2 + model->params.fc.input_size*fc1_size;
    init_params(model->params.output.weights, fc1_size*outputSize);

    model->data = (float*)malloc(imageSize*imageSize*BATCH*sizeof(float));
}

void conv_forward(float *inp, int h, int w,int z, float* conv_weights, int kernel_size, int stride, int channel){

}

void cnn_forward(CNN *model, float learn_rate){
    return;
}



int main(int argc, char const *argv[])
{

    clock_t start, end;
    srand(time(NULL));

    DataLoader dataloader;
    dataloader_init(&dataloader, TRAIN_IMG_PATH, TRAIN_LBL_PATH, 1);
    CNN model;
    CNN_init(&model, ImageSize, K1,C1,1,P1,k2,C2,1,P2,FC1_SIZE, OUTPUT_SIZE, BATCH);

    int train_size = (int)(dataloader.nImages*TRAIN_SPLIT);
    int test_size = dataloader.nImages - train_size;

    for (int epoch = 0; epoch < EPOCHS; epoch++){
        start = clock();
        for(int b=0;b<train_size/BATCH;b++){
            // float* images = dataloader.images + b*BATCH*ImageSize*ImageSize;
            load_betch_images(&dataloader, model.data, b, BATCH);
            float loss = 0.0f;
            for (int t = 0; t < BATCH; t++){
                cnn_forward(&model, LEARN_RATE);
            }
            
        }

    }
    


    free(dataloader.images);
    free(dataloader.labels);
    free(model.params_memory);
    return 0;
}
