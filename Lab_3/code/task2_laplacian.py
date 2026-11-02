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
    "task2"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

# =========================================================
# LOAD IMAGES
# =========================================================

blurred = cv2.imread(
    os.path.join(
        DATASET_DIR,
        "agv_motion_blur.png"
    ),
    cv2.IMREAD_GRAYSCALE
)

clean = cv2.imread(
    os.path.join(
        DATASET_DIR,
        "agv_clean.png"
    ),
    cv2.IMREAD_GRAYSCALE
)


if blurred is None:
    raise FileNotFoundError(
        "Motion blurred image not found."
    )

if clean is None:
    raise FileNotFoundError(
        "Clean image not found."
    )


print("Images loaded successfully.")


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

    output = np.zeros(
        image.shape,
        dtype=np.float32
    )

    # Move kernel across image
    for i in range(image_height):

        for j in range(image_width):

            region = padded[
                i:i + kernel_height,
                j:j + kernel_width
            ]

            output[i, j] = np.sum(
                region * kernel
            )

    return output


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

    return 10 * np.log10(
        (255 ** 2) / mse
    )


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
# LAPLACIAN KERNELS
# =========================================================

laplacian_4 = np.array(
    [
        [0, -1, 0],
        [-1, 4, -1],
        [0, -1, 0]
    ],
    dtype=np.float32
)


laplacian_8 = np.array(
    [
        [-1, -1, -1],
        [-1, 8, -1],
        [-1, -1, -1]
    ],
    dtype=np.float32
)


# =========================================================
# APPLY LAPLACIAN SHARPENING
# =========================================================

def laplacian_sharpen(
    image,
    kernel
):

    # Calculate Laplacian response
    laplacian = correlation(
        image,
        kernel
    )

    # Sharpen the image
    sharpened = (
        image.astype(np.float32)
        +
        laplacian
    )

    # Keep values in valid image range
    sharpened = np.clip(
        sharpened,
        0,
        255
    )

    return sharpened.astype(
        np.uint8
    )


# =========================================================
# 4-NEIGHBOR LAPLACIAN
# =========================================================

result_4 = laplacian_sharpen(
    blurred,
    laplacian_4
)

output_4_path = os.path.join(
    OUTPUT_DIR,
    "laplacian_4_neighbor.png"
)

cv2.imwrite(
    output_4_path,
    result_4
)


# =========================================================
# 8-NEIGHBOR LAPLACIAN
# =========================================================

result_8 = laplacian_sharpen(
    blurred,
    laplacian_8
)

output_8_path = os.path.join(
    OUTPUT_DIR,
    "laplacian_8_neighbor.png"
)

cv2.imwrite(
    output_8_path,
    result_8
)


# =========================================================
# EVALUATION
# =========================================================

psnr_blurred = calculate_psnr(
    clean,
    blurred
)

sharpness_blurred = calculate_sharpness(
    blurred
)

psnr_4 = calculate_psnr(
    clean,
    result_4
)

sharpness_4 = calculate_sharpness(
    result_4
)

psnr_8 = calculate_psnr(
    clean,
    result_8
)

sharpness_8 = calculate_sharpness(
    result_8
)


# =========================================================
# DISPLAY RESULTS
# =========================================================

print("\n==========================================")
print("TASK 2 - LAPLACIAN SHARPENING")
print("==========================================")

print(
    f"\nMotion blurred"
    f" | PSNR = {psnr_blurred:.2f} dB"
    f" | Sharpness = {sharpness_blurred:.2f}"
)

print(
    f"4-neighbor Laplacian"
    f" | PSNR = {psnr_4:.2f} dB"
    f" | Sharpness = {sharpness_4:.2f}"
)

print(
    f"8-neighbor Laplacian"
    f" | PSNR = {psnr_8:.2f} dB"
    f" | Sharpness = {sharpness_8:.2f}"
)


print("\nOutput files:")

print(output_4_path)
print(output_8_path)

print("\nTask 2 completed.")