
import cv2
import numpy as np
import os

# =====================================
# 1. PROJECT PATHS
# =====================================

PROJECT_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)

DATASET_DIR = os.path.join(
    PROJECT_DIR, "Dataset", "Lab_3"
)

OUTPUT_DIR = os.path.join(
    PROJECT_DIR, "Lab_3", "output", "task5"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =====================================
# 2. LOAD IMAGES
# =====================================

clean = cv2.imread(
    os.path.join(DATASET_DIR, "agv_clean.png"),
    cv2.IMREAD_GRAYSCALE
)

noisy_images = {
    10: cv2.imread(
        os.path.join(DATASET_DIR, "agv_noisy_sigma10.png"),
        cv2.IMREAD_GRAYSCALE
    ),
    25: cv2.imread(
        os.path.join(DATASET_DIR, "agv_noisy_sigma25.png"),
        cv2.IMREAD_GRAYSCALE
    )
}

if clean is None:
    raise FileNotFoundError("Clean image not found.")

if any(img is None for img in noisy_images.values()):
    raise FileNotFoundError("Noisy images not found.")


# =====================================
# 3. CREATE 3x3 AVERAGING KERNEL
# =====================================

kernel = np.ones(
    (3, 3),
    dtype=np.float32
) / 9


# =====================================
# 4. HIGH-BOOST PARAMETERS
# =====================================

k = 1.5


# =====================================
# 5. PROCESS BOTH NOISE LEVELS
# =====================================

print("\nTASK 5: FINAL IMAGE PROCESSING PIPELINE")

for sigma, noisy in noisy_images.items():

    # STEP 1: DENOISING

    denoised = cv2.filter2D(
        noisy,
        -1,
        kernel,
        borderType=cv2.BORDER_CONSTANT
    )

    # STEP 2: CREATE BLURRED IMAGE

    smooth = cv2.GaussianBlur(
        denoised,
        (5, 5),
        1.0
    )

    # STEP 3: HIGH-FREQUENCY MASK

    mask = (
        denoised.astype(np.float32)
        - smooth.astype(np.float32)
    )

    # STEP 4: HIGH-BOOST SHARPENING

    final_image = (
        denoised.astype(np.float32)
        + k * mask
    )

    final_image = np.clip(
        final_image, 0, 255
    ).astype(np.uint8)

    # =================================
    # SAVE OUTPUTS
    # =================================

    cv2.imwrite(
        os.path.join(
            OUTPUT_DIR,
            f"sigma{sigma}_denoised.png"
        ),
        denoised
    )

    cv2.imwrite(
        os.path.join(
            OUTPUT_DIR,
            f"sigma{sigma}_final.png"
        ),
        final_image
    )

    # =================================
    # EVALUATION
    # =================================

    mse = np.mean(
        (
            clean.astype(np.float64)
            - final_image.astype(np.float64)
        ) ** 2
    )

    psnr = (
        float("inf")
        if mse == 0
        else 10 * np.log10((255 ** 2) / mse)
    )

    laplacian = cv2.Laplacian(
        final_image,
        cv2.CV_64F
    )

    sharpness = laplacian.var()

    print(
        f"Sigma = {sigma} | "
        f"PSNR = {psnr:.2f} dB | "
        f"Sharpness = {sharpness:.2f}"
    )


print("\nTask 5 completed successfully!")
print("Output folder:", OUTPUT_DIR)
