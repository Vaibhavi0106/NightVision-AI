// CUDA Kernel for Low-Light Gamma Correction
// Normalizes pixel values to [0.0, 1.0], applies non-linear gamma curve: S = I^(1/gamma),
// and maps back to [0, 255] with safe bounds clamping.

extern "C" __global__
void gamma_correction_kernel(
    const unsigned char* __restrict__ src,
    unsigned char* __restrict__ dst,
    int width,
    int height,
    int channels,
    float gamma
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int pixel_idx = (y * width + x) * channels;
        float inv_gamma = 1.0f / gamma;

        for (int c = 0; c < channels; ++c) {
            float norm = (float)src[pixel_idx + c] / 255.0f;
            float corrected = __powf(norm, inv_gamma);
            int val = (int)(corrected * 255.0f + 0.5f);
            if (val < 0) val = 0;
            if (val > 255) val = 255;
            dst[pixel_idx + c] = (unsigned char)val;
        }
    }
}
