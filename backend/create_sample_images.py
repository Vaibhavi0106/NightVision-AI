import os
import cv2
import numpy as np

def generate_sample_images():
    os.makedirs("test_images", exist_ok=True)

    h, w = 720, 1280
    img = np.zeros((h, w, 3), dtype=np.uint8)

    # Add subtle background gradient
    for y in range(h):
        img[y, :, 0] = int((y / h) * 15.0)  # Dark blue tint
        img[y, :, 1] = int((y / h) * 10.0)
        img[y, :, 2] = int((y / h) * 20.0)

    # Draw dark geometric objects & text (simulating dark photo)
    cv2.circle(img, (300, 360), 120, (35, 25, 15), -1)
    cv2.rectangle(img, (600, 200), (950, 550), (15, 35, 25), -1)
    
    cv2.putText(img, "NIGHTVISION AI - LOW LIGHT TEST", (100, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (65, 45, 35), 2)
    cv2.putText(img, "CUDA GPU Acceleration Test Pattern", (100, 600), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (40, 30, 20), 2)

    # Add Gaussian noise
    noise = np.random.normal(0, 4.0, img.shape).astype(np.int16)
    img_noisy = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    sample_path = os.path.join("test_images", "sample_dark.jpg")
    cv2.imwrite(sample_path, img_noisy)
    print(f"Sample low-light image saved to: {sample_path}")

if __name__ == "__main__":
    generate_sample_images()
