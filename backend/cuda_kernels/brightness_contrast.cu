// CUDA Kernel for Brightness, Contrast & Exposure Adjustment
// Formula: output = contrast * (input - 128) + 128 + brightness
// Values are clamped to [0, 255] safely.

extern "C" __global__
void brightness_contrast_kernel(
    const unsigned char* __restrict__ src,
    unsigned char* __restrict__ dst,
    int width,
    int height,
    int channels,
    float brightness,
    float contrast
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int pixel_idx = (y * width + x) * channels;

        for (int c = 0; c < channels; ++c) {
            float val = (float)src[pixel_idx + c];
            float res = contrast * (val - 128.0f) + 128.0f + brightness;
            int out_val = (int)(res + 0.5f);
            if (out_val < 0) out_val = 0;
            if (out_val > 255) out_val = 255;
            dst[pixel_idx + c] = (unsigned char)out_val;
        }
    }
}
