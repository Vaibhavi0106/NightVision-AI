// Combined Low-Light Auto Enhancement CUDA Kernel
// Merges Gamma Correction, Brightness, Contrast & Sharpening into a single GPU pass.
// Reuses pixel registers in GPU threads to avoid memory roundtrips.

extern "C" __global__
void combined_auto_enhance_v2(
    const unsigned char* __restrict__ src,
    unsigned char* __restrict__ dst,
    int width,
    int height,
    int channels,
    float gamma,
    float brightness,
    float contrast,
    float sharpen_strength
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        float inv_gamma = 1.0f / gamma;
        int center_idx = (y * width + x) * channels;

        for (int c = 0; c < channels; ++c) {
            #define TRANSFORM_AND_CLAMP(idx) \
                fminf(fmaxf(contrast * ((__powf((float)src[(idx) + c] / 255.0f, inv_gamma) * 255.0f) - 128.0f) + 128.0f + brightness, 0.0f), 255.0f)

            float center_cb = TRANSFORM_AND_CLAMP(center_idx);

            float final_val = center_cb;
            if (sharpen_strength > 0.001f) {
                int top_y    = max(y - 1, 0);
                int bottom_y = min(y + 1, height - 1);
                int left_x   = max(x - 1, 0);
                int right_x  = min(x + 1, width - 1);

                int top_idx    = (top_y * width + x) * channels;
                int bottom_idx = (bottom_y * width + x) * channels;
                int left_idx   = (y * width + left_x) * channels;
                int right_idx  = (y * width + right_x) * channels;

                float top_cb    = TRANSFORM_AND_CLAMP(top_idx);
                float bottom_cb = TRANSFORM_AND_CLAMP(bottom_idx);
                float left_cb   = TRANSFORM_AND_CLAMP(left_idx);
                float right_cb  = TRANSFORM_AND_CLAMP(right_idx);

                float lap = 4.0f * center_cb - (top_cb + bottom_cb + left_cb + right_cb);
                final_val = center_cb + sharpen_strength * lap;
            }

            #undef TRANSFORM_AND_CLAMP

            int out_int = (int)(final_val + 0.5f);
            if (out_int < 0) out_int = 0;
            if (out_int > 255) out_int = 255;

            dst[center_idx + c] = (unsigned char)out_int;
        }
    }
}
