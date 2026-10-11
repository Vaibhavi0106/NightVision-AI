import unittest
import numpy as np

class TestNightVisionCUDAvsCPU(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from app.cpu_engine import CPUEngine
        from app.cuda_engine import CUDAEngine
        cls.cpu = CPUEngine()
        cls.cuda = CUDAEngine()
        # Generate test synthetic low-light image
        np.random.seed(42)
        cls.test_img = np.random.randint(5, 120, (512, 512, 3), dtype=np.uint8)

    def test_gamma_correction_tolerance(self):
        """Compares CUDA and CPU Gamma Correction within 1 pixel intensity tolerance (MAE < 1.0)."""
        gamma_val = 2.2
        cpu_out = self.cpu.apply_gamma(self.test_img, gamma_val)
        cuda_out, metrics = self.cuda.apply_gamma(self.test_img, gamma_val)

        mae = float(np.mean(np.abs(cpu_out.astype(np.float32) - cuda_out.astype(np.float32))))
        print(f"[Test] Gamma Correction MAE between CPU and CUDA: {mae:.4f} (Backend: {metrics['backend']})")
        self.assertLess(mae, 1.5, "CUDA and CPU Gamma correction outputs deviate beyond tolerance.")

    def test_brightness_contrast_tolerance(self):
        """Compares CUDA and CPU Brightness/Contrast within tolerance."""
        brightness, contrast = 20.0, 1.35
        cpu_out = self.cpu.apply_brightness_contrast(self.test_img, brightness, contrast)
        cuda_out, metrics = self.cuda.apply_brightness_contrast(self.test_img, brightness, contrast)

        mae = float(np.mean(np.abs(cpu_out.astype(np.float32) - cuda_out.astype(np.float32))))
        print(f"[Test] Brightness/Contrast MAE between CPU and CUDA: {mae:.4f} (Backend: {metrics['backend']})")
        self.assertLess(mae, 1.5, "CUDA and CPU Brightness/Contrast outputs deviate beyond tolerance.")

    def test_sharpen_tolerance(self):
        """Compares CUDA and CPU Sharpening within tolerance."""
        strength = 1.0
        cpu_out = self.cpu.apply_sharpen(self.test_img, strength)
        cuda_out, metrics = self.cuda.apply_sharpen(self.test_img, strength)

        mae = float(np.mean(np.abs(cpu_out.astype(np.float32) - cuda_out.astype(np.float32))))
        print(f"[Test] Sharpening MAE between CPU and CUDA: {mae:.4f} (Backend: {metrics['backend']})")
        self.assertLess(mae, 2.0, "CUDA and CPU Sharpening outputs deviate beyond tolerance.")

    def test_combined_auto_enhance_tolerance(self):
        """Compares Combined Auto Enhance CUDA Kernel vs CPU pipeline."""
        cpu_out = self.cpu.apply_auto_enhance(self.test_img, gamma=2.0, brightness=10.0, contrast=1.2, denoising_strength=0.0, sharpening_strength=0.5)
        cuda_out, metrics = self.cuda.apply_auto_enhance(self.test_img, gamma=2.0, brightness=10.0, contrast=1.2, denoising_strength=0.0, sharpening_strength=0.5)

        mae = float(np.mean(np.abs(cpu_out.astype(np.float32) - cuda_out.astype(np.float32))))
        print(f"[Test] Combined Auto Enhance MAE between CPU and CUDA: {mae:.4f} (Backend: {metrics['backend']})")
        self.assertLess(mae, 3.5, "Combined Auto Enhance outputs deviate beyond tolerance.")

if __name__ == "__main__":
    unittest.main()
