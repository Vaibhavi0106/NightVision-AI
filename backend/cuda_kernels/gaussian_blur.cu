// CUDA Kernels for Gaussian Blur Noise Reduction
// Includes:
// 1. Global memory baseline convolution kernel.
// 2. Shared-memory tiled convolution kernel for optimized memory bandwidth.

#define BLOCK_DIM_X 16
#define BLOCK_DIM_Y 16
#define MAX_KERNEL_RADIUS 5
#define TILE_WIDTH (BLOCK_DIM_X + 2 * MAX_KERNEL_RADIUS)
#define TILE_HEIGHT (BLOCK_DIM_Y + 2 * MAX_KERNEL_RADIUS)

// Global memory Gaussian blur kernel (5x5 filter kernel)
extern "C" __global__
void gaussian_blur_global(
    const unsigned char* __restrict__ src,
    unsigned char* __restrict__ dst,
    int width,
    int height,
    int channels,
    const float* __restrict__ kernel,
    int k_size
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int radius = k_size / 2;

        for (int c = 0; c < channels; ++c) {
            float sum = 0.0f;
            float weight_sum = 0.0f;

            for (int ky = -radius; ky <= radius; ++ky) {
                for (int kx = -radius; kx <= radius; ++kx) {
                    int ix = min(max(x + kx, 0), width - 1);
                    int iy = min(max(y + ky, 0), height - 1);
                    float w = kernel[(ky + radius) * k_size + (kx + radius)];

                    sum += (float)src[(iy * width + ix) * channels + c] * w;
                    weight_sum += w;
                }
            }

            int out_val = (int)(sum / weight_sum + 0.5f);
            if (out_val < 0) out_val = 0;
            if (out_val > 255) out_val = 255;
            dst[(y * width + x) * channels + c] = (unsigned char)out_val;
        }
    }
}

// Shared-memory tiled 5x5 Gaussian blur kernel
extern "C" __global__
void gaussian_blur_shared(
    const unsigned char* __restrict__ src,
    unsigned char* __restrict__ dst,
    int width,
    int height,
    int channels,
    const float* __restrict__ kernel,
    int k_size
) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int x = blockIdx.x * blockDim.x + tx;
    int y = blockIdx.y * blockDim.y + ty;

    int radius = k_size / 2;

    // Shared memory for RGB block tile including apron/halo
    __shared__ float s_tile[TILE_HEIGHT][TILE_WIDTH][3];

    // Load halo and main region into shared memory
    int g_x = min(max(x - radius, 0), width - 1);
    int g_y = min(max(y - radius, 0), height - 1);

    for (int c = 0; c < channels && c < 3; ++c) {
        s_tile[ty][tx][c] = (float)src[(g_y * width + g_x) * channels + c];
    }

    // Load right halo if needed
    if (tx < 2 * radius) {
        int halo_x = min(max(x + blockDim.x - radius, 0), width - 1);
        for (int c = 0; c < channels && c < 3; ++c) {
            s_tile[ty][tx + blockDim.x][c] = (float)src[(g_y * width + halo_x) * channels + c];
        }
    }

    // Load bottom halo if needed
    if (ty < 2 * radius) {
        int halo_y = min(max(y + blockDim.y - radius, 0), height - 1);
        for (int c = 0; c < channels && c < 3; ++c) {
            s_tile[ty + blockDim.y][tx][c] = (float)src[(halo_y * width + g_x) * channels + c];
        }
    }

    // Load bottom-right halo corner
    if (tx < 2 * radius && ty < 2 * radius) {
        int halo_x = min(max(x + blockDim.x - radius, 0), width - 1);
        int halo_y = min(max(y + blockDim.y - radius, 0), height - 1);
        for (int c = 0; c < channels && c < 3; ++c) {
            s_tile[ty + blockDim.y][tx + blockDim.x][c] = (float)src[(halo_y * width + halo_x) * channels + c];
        }
    }

    __syncthreads();

    if (x < width && y < height) {
        for (int c = 0; c < channels && c < 3; ++c) {
            float sum = 0.0f;
            float weight_sum = 0.0f;

            for (int ky = 0; ky < k_size; ++ky) {
                for (int kx = 0; kx < k_size; ++kx) {
                    float w = kernel[ky * k_size + kx];
                    sum += s_tile[ty + ky][tx + kx][c] * w;
                    weight_sum += w;
                }
            }

            int out_val = (int)(sum / weight_sum + 0.5f);
            if (out_val < 0) out_val = 0;
            if (out_val > 255) out_val = 255;
            dst[(y * width + x) * channels + c] = (unsigned char)out_val;
        }
    }
}
