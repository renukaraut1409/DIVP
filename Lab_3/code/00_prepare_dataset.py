import cv2
import numpy as np
import os


# =========================================================
# FIND PROJECT DIRECTORIES
# =========================================================

# Current file:
# DIVP/Lab_3/code/00_prepare_dataset.py

PROJECT_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)


# Original image is inside:
# DIVP/Dataset/Lab_3/

DATASET_DIR = os.path.join(
    PROJECT_DIR,
    "Dataset",
    "Lab_3"
)


# =========================================================
# FILE PATHS
# =========================================================

input_path = os.path.join(
    DATASET_DIR,
    "agv_clean.JPG"
)

clean_gray_path = os.path.join(
    DATASET_DIR,
    "agv_clean.png"
)

noise10_path = os.path.join(
    DATASET_DIR,
    "agv_noisy_sigma10.png"
)

noise25_path = os.path.join(
    DATASET_DIR,
    "agv_noisy_sigma25.png"
)

motion_path = os.path.join(
    DATASET_DIR,
    "agv_motion_blur.png"
)


# =========================================================
# CHECK DATASET FOLDER
# =========================================================

if not os.path.exists(DATASET_DIR):
    raise FileNotFoundError(
        f"Dataset folder not found:\n{DATASET_DIR}"
    )


# =========================================================
# LOAD ORIGINAL IMAGE
# =========================================================

image = cv2.imread(input_path)

if image is None:
    raise FileNotFoundError(
        f"Could not load image:\n{input_path}"
    )


print("Original image loaded successfully.")
print("Image size:", image.shape)


# =========================================================
# CONVERT TO GRAYSCALE
# =========================================================

gray = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2GRAY
)

cv2.imwrite(
    clean_gray_path,
    gray
)

print("Grayscale image saved:")
print(clean_gray_path)


# =========================================================
# FUNCTION TO ADD GAUSSIAN NOISE
# =========================================================

def add_gaussian_noise(image, sigma):

    noise = np.random.normal(
        loc=0,
        scale=sigma,
        size=image.shape
    )

    noisy = image.astype(
        np.float32
    ) + noise

    noisy = np.clip(
        noisy,
        0,
        255
    )

    return noisy.astype(
        np.uint8
    )


# =========================================================
# GENERATE GAUSSIAN NOISE
# =========================================================

np.random.seed(42)

noisy_sigma10 = add_gaussian_noise(
    gray,
    10
)

noisy_sigma25 = add_gaussian_noise(
    gray,
    25
)


cv2.imwrite(
    noise10_path,
    noisy_sigma10
)

cv2.imwrite(
    noise25_path,
    noisy_sigma25
)

print("Sigma = 10 noisy image saved.")
print("Sigma = 25 noisy image saved.")


# =========================================================
# GENERATE MOTION BLUR
# =========================================================

kernel_size = 15

motion_kernel = np.zeros(
    (kernel_size, kernel_size),
    dtype=np.float32
)

motion_kernel[
    kernel_size // 2,
    :
] = 1.0 / kernel_size


motion_blur = cv2.filter2D(
    gray,
    -1,
    motion_kernel
)


cv2.imwrite(
    motion_path,
    motion_blur
)

print("Motion blurred image saved.")


# =========================================================
# COMPLETION
# =========================================================

print("\nDataset preparation completed successfully.")

print("\nGenerated files:")

print("1.", clean_gray_path)
print("2.", noise10_path)
print("3.", noise25_path)
print("4.", motion_path)