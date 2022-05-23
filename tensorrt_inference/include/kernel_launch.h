#define WIDTH 1024
#define HEIGHT 512
#define CHANNELS 19
#define DONT_CARE -1

void argmaxLaunchKernel(float *bisenet_output, float *argmax_output, cudaStream_t stream);