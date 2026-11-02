import cv2
import numpy as np
import os


PROJECT_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

DATASET_DIR = os.path.join(
    PROJECT_DIR,
    "Dataset",
    "Lab_3"
)

OUTPUT_DIR = os.path.join(
    PROJECT_DIR,
    "Lab_3",
    "output",
    "task1"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# =========================================================
# LOAD INPUT IMAGES
# =========================================================

clean = cv2.imread(
    os.path.join(
        DATASET_DIR,
        "agv_clean.png"
    ),
    cv2.IMREAD_GRAYSCALE
)

noisy_sigma10 = cv2.imread(
    os.path.join(
        DATASET_DIR,
        "agv_noisy_sigma10.png"
    ),
    cv2.IMREAD_GRAYSCALE
)

noisy_sigma25 = cv2.imread(
    os.path.join(
        DATASET_DIR,
        "agv_noisy_sigma25.png"
    ),
    cv2.IMREAD_GRAYSCALE
)


if clean is None:
    raise FileNotFoundError("Clean image not found.")

if noisy_sigma10 is None:
    raise FileNotFoundError("Sigma 10 image not found.")

if noisy_sigma25 is None:
    raise FileNotFoundError("Sigma 25 image not found.")


print("Images loaded successfully.")


# =========================================================
# CREATE AVERAGING KERNEL
# =========================================================

def create_averaging_kernel(size):

    kernel = np.ones(
        (size, size),
        dtype=np.float32
    )

    kernel = kernel / (size * size)

    return kernel


# =========================================================
# CUSTOM CORRELATION FUNCTION
# =========================================================

def correlation(image, kernel):

    image_height, image_width = image.shape

    kernel_height, kernel_width = kernel.shape

    pad_h = kernel_height // 2
    pad_w = kernel_width // 2

    # Zero padding
    padded = np.pad(
        image,
        (
            (pad_h, pad_h),
            (pad_w, pad_w)
        ),
        mode="constant"
    )

    output = np.zeros_like(
        image,
        dtype=np.float32
    )

    # Move kernel over every pixel
    for i in range(image_height):

        for j in range(image_width):

            region = padded[
                i:i + kernel_height,
                j:j + kernel_width
            ]

            output[i, j] = np.sum(
                region * kernel
            )

    # Keep values between 0 and 255
    output = np.clip(
        output,
        0,
        255
    )

    return output.astype(
        np.uint8
    )


# =========================================================
# PSNR
# =========================================================

def calculate_psnr(
    reference,
    processed
):

    mse = np.mean(
        (
            reference.astype(np.float64)
            -
            processed.astype(np.float64)
        ) ** 2
    )

    if mse == 0:
        return float("inf")

    psnr = 10 * np.log10(
        (255 ** 2) / mse
    )

    return psnr


# =========================================================
# SHARPNESS
# =========================================================

def calculate_sharpness(image):

    laplacian = cv2.Laplacian(
        image,
        cv2.CV_64F
    )

    return laplacian.var()


# =========================================================
# EXPERIMENT
# =========================================================

kernel_sizes = [3, 5, 9]

noisy_images = {
    10: noisy_sigma10,
    25: noisy_sigma25
}


print("\n==============================")
print("TASK 1: AVERAGING FILTER")
print("==============================")

for sigma, noisy_image in noisy_images.items():

    print(
        f"\nNoise level: sigma = {sigma}"
    )

    for size in kernel_sizes:

        # Create averaging kernel
        kernel = create_averaging_kernel(
            size
        )

        # Apply our custom correlation
        filtered = correlation(
            noisy_image,
            kernel
        )

        # Save result
        output_name = (
            f"sigma{sigma}_"
            f"{size}x{size}.png"
        )

        output_path = os.path.join(
            OUTPUT_DIR,
            output_name
        )

        cv2.imwrite(
            output_path,
            filtered
        )

        # Calculate metrics
        psnr = calculate_psnr(
            clean,
            filtered
        )

        sharpness = calculate_sharpness(
            filtered
        )

        print(
            f"{size}x{size} "
            f"| PSNR = {psnr:.2f} dB "
            f"| Sharpness = {sharpness:.2f}"
        )


print("\nTask 1 completed.")

print(
    "\nOutput images saved in:"
)

print(OUTPUT_DIR)