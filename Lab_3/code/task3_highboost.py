
import cv2
import numpy as np
import os

# PROJECT PATHS
PROJECT_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)

DATASET_DIR = os.path.join(PROJECT_DIR, "Dataset", "Lab_3")

OUTPUT_DIR = os.path.join(
    PROJECT_DIR, "Lab_3", "output", "task3"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# LOAD MOTION-BLURRED IMAGE
image = cv2.imread(
    os.path.join(DATASET_DIR, "agv_motion_blur.png"),
    cv2.IMREAD_GRAYSCALE
)

if image is None:
    raise FileNotFoundError("Motion blurred image not found.")

# GAUSSIAN BLUR
blurred = cv2.GaussianBlur(image, (5, 5), 1.0)

# HIGH-BOOST FILTERING
k_values = [1, 1.5, 2, 3]

print("\nTASK 3: HIGH-BOOST SHARPENING")

for k in k_values:

    # Calculate high-frequency mask
    mask = image.astype(np.float32) - blurred.astype(np.float32)

    # Add amplified details
    sharpened = image.astype(np.float32) + k * mask

    # Keep valid pixel intensity range
    sharpened = np.clip(sharpened, 0, 255)

    sharpened = sharpened.astype(np.uint8)

    # Save output
    filename = f"highboost_k{k}.png"

    output_path = os.path.join(OUTPUT_DIR, filename)

    cv2.imwrite(output_path, sharpened)

    print(f"k = {k}: Saved {filename}")

print("\nTask 3 completed.")
print("Output folder:", OUTPUT_DIR)
