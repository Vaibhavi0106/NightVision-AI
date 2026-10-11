import time
import cv2
import numpy as np

class CPUEngine:
    """
    CPU reference implementation for low-light photo and video enhancement algorithms.
    Provides mathematically exact reference results for performance comparisons and CPU fallback mode.
    """

    @staticmethod
    def apply_gamma(img: np.ndarray, gamma: float) -> np.ndarray:
        """Applies gamma correction using an 8-bit Lookup Table (LUT)."""
        if gamma <= 0:
            gamma = 1.0
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
        return cv2.LUT(img, table)

    @staticmethod
    def apply_brightness_contrast(img: np.ndarray, brightness: float, contrast: float) -> np.ndarray:
        """
        Adjusts brightness and contrast.
        brightness: [-100, 100], contrast: [0.1, 3.0]
        Formula: output = contrast * (input - 128) + 128 + brightness
        """
        img_float = img.astype(np.float32)
        res = contrast * (img_float - 128.0) + 128.0 + brightness
        return np.clip(res, 0, 255).astype(np.uint8)

    @staticmethod
    def apply_auto_brightness(img: np.ndarray) -> tuple[np.ndarray, float, float]:
        """
        Analyzes image brightness and calculates optimal gamma and contrast parameters.
        Prevents highlight clipping. Returns (enhanced_img, calculated_gamma, calculated_contrast).
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        mean_lum = float(np.mean(gray))

        target_lum = 120.0
        if mean_lum < 1.0:
            mean_lum = 1.0
        
        calculated_gamma = float(np.clip(np.log(target_lum / 255.0) / np.log(mean_lum / 255.0), 1.2, 3.5))
        calculated_contrast = float(np.clip(1.0 + (target_lum - mean_lum) / 255.0, 1.0, 1.6))

        enhanced = CPUEngine.apply_gamma(img, calculated_gamma)
        enhanced = CPUEngine.apply_brightness_contrast(enhanced, brightness=5.0, contrast=calculated_contrast)
        return enhanced, calculated_gamma, calculated_contrast

    @staticmethod
    def apply_histogram_equalization(img: np.ndarray, use_clahe: bool = True, clip_limit: float = 2.0) -> np.ndarray:
        """
        Applies Histogram Equalization or CLAHE in YCrCb color space to preserve color balance.
        """
        if len(img.shape) == 2:
            if use_clahe:
                clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
                return clahe.apply(img)
            else:
                return cv2.equalizeHist(img)
        else:
            ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
            y, cr, cb = cv2.split(ycrcb)
            if use_clahe:
                clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
                y_eq = clahe.apply(y)
            else:
                y_eq = cv2.equalizeHist(y)
            merged = cv2.merge([y_eq, cr, cb])
            return cv2.cvtColor(merged, cv2.COLOR_YCrCb2BGR)

    @staticmethod
    def apply_gaussian_blur(img: np.ndarray, ksize: int = 5, sigma: float = 1.0) -> np.ndarray:
        """Applies 2D Gaussian blur for noise reduction."""
        if ksize % 2 == 0:
            ksize += 1
        return cv2.GaussianBlur(img, (ksize, ksize), sigma)

    @staticmethod
    def apply_median_filter(img: np.ndarray, ksize: int = 5) -> np.ndarray:
        """Applies Median filter noise reduction."""
        if ksize % 2 == 0:
            ksize += 1
        return cv2.medianBlur(img, ksize)

    @staticmethod
    def apply_sharpen(img: np.ndarray, strength: float = 1.0) -> np.ndarray:
        """Applies 3x3 Laplacian sharpening filter (Unsharp masking)."""
        if strength <= 0.0:
            return img.copy()
        
        # Unsharp masking filter weights:
        # [  0         -strength      0       ]
        # [-strength  1+4*strength  -strength ]
        # [  0         -strength      0       ]
        s = float(strength)
        kernel = np.array([[ 0, -s,  0],
                           [-s, 1.0 + 4.0 * s, -s],
                           [ 0, -s,  0]], dtype=np.float32)
        sharpened = cv2.filter2D(img, -1, kernel)
        return np.clip(sharpened, 0, 255).astype(np.uint8)

    @staticmethod
    def apply_auto_enhance(
        img: np.ndarray,
        gamma: float = 2.0,
        brightness: float = 10.0,
        contrast: float = 1.25,
        denoising_strength: float = 0.5,
        sharpening_strength: float = 0.5,
        use_clahe: bool = False
    ) -> np.ndarray:
        """Executes full CPU low-light enhancement pipeline."""
        out = img.copy()
        if gamma != 1.0:
            out = CPUEngine.apply_gamma(out, gamma)
        if brightness != 0.0 or contrast != 1.0:
            out = CPUEngine.apply_brightness_contrast(out, brightness, contrast)
        if use_clahe:
            out = CPUEngine.apply_histogram_equalization(out, use_clahe=True)
        if denoising_strength > 0.0:
            ksize = 3 if denoising_strength < 0.5 else 5
            out = CPUEngine.apply_gaussian_blur(out, ksize=ksize, sigma=denoising_strength * 1.5)
        if sharpening_strength > 0.0:
            out = CPUEngine.apply_sharpen(out, strength=sharpening_strength)
        return out
