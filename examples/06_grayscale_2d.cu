// 06 — A 2D kernel: convert an RGB image to grayscale. This program is correct.
//
// The launch uses a 2D grid of 2D blocks, so each GPU thread gets an (x, y)
// pixel position. Open the visualization to see the blocks tile the image.
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#define W 40
#define H 24

__global__ void toGray(const unsigned char *rgb, unsigned char *gray, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        int p = y * width + x;
        unsigned char r = rgb[3 * p];
        unsigned char g = rgb[3 * p + 1];
        unsigned char b = rgb[3 * p + 2];
        gray[p] = (unsigned char)(0.299f * r + 0.587f * g + 0.114f * b);
    }
}

int main(void)
{
    unsigned char *rgb = (unsigned char *)malloc(3 * W * H);
    unsigned char *gray = (unsigned char *)malloc(W * H);
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++) {
            int p = y * W + x;
            rgb[3 * p] = (unsigned char)(255 * x / (W - 1));
            rgb[3 * p + 1] = (unsigned char)(255 * y / (H - 1));
            rgb[3 * p + 2] = 128;
        }

    unsigned char *d_rgb, *d_gray;
    cudaMalloc((void **)&d_rgb, 3 * W * H);
    cudaMalloc((void **)&d_gray, W * H);
    cudaMemcpy(d_rgb, rgb, 3 * W * H, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(8, 8);
    dim3 numBlocks((W + 7) / 8, (H + 7) / 8);
    toGray<<<numBlocks, threadsPerBlock>>>(d_rgb, d_gray, W, H);
    cudaDeviceSynchronize();

    cudaMemcpy(gray, d_gray, W * H, cudaMemcpyDeviceToHost);
    for (int y = 0; y < H; y += 4) {
        for (int x = 0; x < W; x++) putchar(" .:-=+*#%@"[gray[y * W + x] * 10 / 256]);
        putchar('\n');
    }

    cudaFree(d_rgb);
    cudaFree(d_gray);
    free(rgb);
    free(gray);
    return 0;
}
