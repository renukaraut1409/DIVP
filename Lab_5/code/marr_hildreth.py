
# -*- coding: utf-8 -*-
"""Marr-Hildreth Edge Detection - DIVP Lab 5"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ==================================================
# INPUT AND OUTPUT PATHS
# ==================================================

PROJECT_DIR = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_DIR / "Dataset" / "Lab_5" / "input_image.png"
)

OUTPUT_DIR = (
    PROJECT_DIR / "Lab_5" / "output"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ==================================================
# 1. LOAD IMAGE
# ==================================================

img = cv2.imread(str(INPUT_PATH), cv2.IMREAD_GRAYSCALE)

if img is None:
    raise ValueError(f"Unable to read image: {INPUT_PATH}")

# Scale intensity values to [0, 1]
img = img.astype(np.float64) / 255.0

print("Image size:", img.shape)
print("Intensity range:", img.min(), "to", img.max())

# ==================================================
# 2. GAUSSIAN SMOOTHING
# ==================================================

# sigma = 4
# n = 25
# sigma = 7
# n = 35

sigma = 9
n = 45

smoothed = cv2.GaussianBlur(
    img,
    (n, n),
    sigmaX=sigma,
    sigmaY=sigma
)

# ==================================================
# 3. LAPLACIAN OF THE SMOOTHED IMAGE
# ==================================================

laplacian = cv2.Laplacian(
    smoothed,
    cv2.CV_64F,
    ksize=3
)

# ==================================================
# 4. ZERO-CROSSING DETECTION
# ==================================================

def find_zero_crossings(L, threshold):

    edges = np.zeros(L.shape, dtype=np.uint8)

    # Opposing neighbor directions
    directions = [
        (0, 1),    # Left-right
        (1, 0),    # Up-down
        (1, 1),    # Diagonal
        (1, -1)    # Other diagonal
    ]

    rows, cols = L.shape

    for dy, dx in directions:

        y1 = max(0, -dy)
        y2 = min(rows, rows - dy)

        x1 = max(0, -dx)
        x2 = min(cols, cols - dx)

        A = L[y1:y2, x1:x2]
        B = L[y1+dy:y2+dy, x1+dx:x2+dx]

        # Opposite signs indicate a zero crossing
        crossing = (
            ((A < 0) & (B >= 0)) |
            ((A >= 0) & (B < 0))
        )

        # Threshold on the response difference
        crossing &= (np.abs(A - B) > threshold)

        edges[y1:y2, x1:x2][crossing] = 255

        edges[y1+dy:y2+dy, x1+dx:x2+dx][crossing] = 255

    return edges


# Threshold = 0
edges_T0 = find_zero_crossings(
    laplacian,
    threshold=0
)

# Threshold = 4% of maximum absolute response
T4 = 0.04 * np.max(np.abs(laplacian))

edges_T4 = find_zero_crossings(
    laplacian,
    threshold=T4
)

print("4% threshold:", T4)

# ==================================================
# 5. DISPLAY ALL RESULTS (2x3 GRID)
# ==================================================

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# (a) Original Image
axes[0, 0].imshow(
    img,
    cmap="gray",
    vmin=0,
    vmax=1
)
axes[0, 0].set_title("(a) Original Image")

# (b) Smoothed Image
axes[0, 1].imshow(
    smoothed,
    cmap="gray",
    vmin=0,
    vmax=1
)
axes[0, 1].set_title(
    f"(b) Smoothed Image (sigma={sigma})"
)

# (c) Laplacian Magnitude
axes[0, 2].imshow(
    np.abs(laplacian),
    cmap="gray"
)
axes[0, 2].set_title("(c) Laplacian Magnitude")

# (d) Signed Laplacian
limit = np.max(np.abs(laplacian))

axes[1, 0].imshow(
    laplacian,
    cmap="gray",
    vmin=-limit,
    vmax=limit
)
axes[1, 0].set_title("(d) Gaussian + Laplacian")

# (e) Zero Crossings T = 0
axes[1, 1].imshow(
    edges_T0,
    cmap="gray",
    vmin=0,
    vmax=255
)
axes[1, 1].set_title("(e) Zero Crossings, T = 0")

# (f) Zero Crossings T = 4%
axes[1, 2].imshow(
    edges_T4,
    cmap="gray",
    vmin=0,
    vmax=255
)
axes[1, 2].set_title(
    "(f) Zero Crossings, T = 4% Max"
)

for ax in axes.ravel():
    ax.axis("off")

plt.suptitle(
    f"Marr-Hildreth Edge Detection "
    f"(Sigma = {sigma}, Kernel = {n}x{n})",
    fontsize=15
)

plt.tight_layout()

# ==================================================
# 6. SAVE FINAL OUTPUT
# ==================================================

output_file = OUTPUT_DIR / "marr_hildreth_output.png"

plt.savefig(
    output_file,
    dpi=150,
    bbox_inches="tight"
)

plt.show()
plt.close(fig)

print("\nMarr-Hildreth completed successfully!")
print("Input image:", INPUT_PATH)
print("Output saved at:", output_file)
