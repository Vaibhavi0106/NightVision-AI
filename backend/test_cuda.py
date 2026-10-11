import os
import time
import numpy as np

def test_cuda():
    print("Checking CUDA environment...")
    try:
        import cupy as cp
        print(f"CuPy Version: {cp.__version__}")
        dev_count = cp.cuda.runtime.getDeviceCount()
        print(f"CUDA Devices Found: {dev_count}")
        if dev_count > 0:
            props = cp.cuda.runtime.getDeviceProperties(0)
            gpu_name = props['name'].decode('utf-8')
            print(f"Active GPU: {gpu_name}")

            # Define CUDA C++ Gamma Kernel
            cuda_source = """
            extern "C" __global__
            void gamma_kernel(const unsigned char* src, unsigned char* dst, int width, int height, int channels, float gamma) {
                int x = blockIdx.x * blockDim.x + threadIdx.x;
                int y = blockIdx.y * blockDim.y + threadIdx.y;

                if (x < width && y < height) {
                    for (int c = 0; c < channels; ++c) {
                        int idx = (y * width + x) * channels + c;
                        float val = (float)src[idx] / 255.0f;
                        float corr = __powf(val, 1.0f / gamma);
                        int out_val = (int)(corr * 255.0f + 0.5f);
                        if (out_val < 0) out_val = 0;
                        if (out_val > 255) out_val = 255;
                        dst[idx] = (unsigned char)out_val;
                    }
                }
            }
            """
            kernel = cp.RawKernel(cuda_source, 'gamma_kernel')
            img = np.random.randint(0, 256, (1080, 1920, 3), dtype=np.uint8)

            d_src = cp.asarray(img)
            d_dst = cp.empty_like(d_src)

            threadsperblock = (16, 16)
            blockspergrid_x = (img.shape[1] + threadsperblock[0] - 1) // threadsperblock[0]
            blockspergrid_y = (img.shape[0] + threadsperblock[1] - 1) // threadsperblock[1]
            blockspergrid = (blockspergrid_x, blockspergrid_y)

            start_evt = cp.cuda.Event()
            end_evt = cp.cuda.Event()

            start_evt.record()
            kernel(blockspergrid, threadsperblock, (d_src, d_dst, img.shape[1], img.shape[0], img.shape[2], np.float32(2.2)))
            end_evt.record()
            end_evt.synchronize()

            ms = cp.cuda.get_elapsed_time(start_evt, end_evt)
            result = cp.asnumpy(d_dst)

            print(f"CUDA Gamma Kernel executed successfully in {ms:.4f} ms!")
            print(f"Input shape: {img.shape}, Output shape: {result.shape}")
            return True
        else:
            print("No CUDA devices detected.")
            return False
    except Exception as e:
        print(f"CUDA test failed: {e}")
        return False

if __name__ == "__main__":
    test_cuda()
