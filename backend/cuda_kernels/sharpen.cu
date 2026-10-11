// CUDA Kernel for Unsharp Masking & Sharpening Convolution
// Uses a 3x3 Laplacian edge enhancement matrix scaled by sharpening strength factor.

extern "C" __global__
void sharpen_kernel(
    const unsigned char* __restrict__ src,
    unsigned char* __restrict__ dst,
    int width,
    int height,
    int channels,
    float strength
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        // 3x3 Laplacian sharpening filter weights:
        // [ 0  -1   0 ]
        // [-1 4+S  -1 ]
        // [ 0  -1   0 ]

        for (int c = 0; c < channels; ++c) {
            float center = (float)src[(y * width + x) * channels + c];

            int top_y    = max(y - 1, 0);
            int bottom_y = min(y + 1, height - 1);
            int left_x   = max(x - 1, 0);
            int right_x  = min(x + 1, width - 1);

            float top    = (float)src[(top_y * width + x) * channels + c];
            float bottom = (float)src[(bottom_y * width + x) * channels + c];
            float left   = (float)src[(y * width + left_x) * channels + c];
            float right  = (float)src[(y * width + right_x) * channels + c];

            float laplacian = 4.0f * center - (top + bottom + left + right);
            float enhanced = center + strength * laplacian;

            int out_val = (int)(enhanced + 0.5f);
            if (out_val < 0) out_val = 0;
            if (out_val > 255) out_val = 255;

            dst[(y * width + x) * channels + c] = (unsigned char)out_val;
        }
    }
}
