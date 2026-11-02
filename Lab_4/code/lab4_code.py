# -*- coding: utf-8 -*-
"""DIVP Lab4.ipynb

Original file is located at
https://colab.research.google.com/drive/18ggA2B8VK10eRkilQXXMe_KyJUwdjR7x
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path

# ==================================================
# PATHS: INPUT IMAGES AND OUTPUT FOLDER
# ==================================================

# This file should be saved in: DIVP/Lab_4/code/
# Project structure:
# DIVP/
#   Dataset/Lab_4/
#       horizontal.png
#       vertical-line.jpg
#       diagonal.jpg
#   Lab_4/
#       code/lab4_complete.py
#       output/

PROJECT_DIR = Path(__file__).resolve().parents[2]
INPUT_DIR = PROJECT_DIR / "Dataset" / "Lab_4"
OUTPUT_DIR = PROJECT_DIR / "Lab_4" / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Given intensity array
Ip = np.array([
    5, 5, 4, 3, 2, 1, 0, 0, 0, 6,
    0, 0, 0, 1, 3, 1, 0, 0, 0, 0,
    7, 7, 5
], dtype=float)

# Method 1: Display the array as a 1-row image
image_1d = Ip.reshape(1, -1)

plt.figure(figsize=(12, 2))
plt.imshow(image_1d, cmap="gray", aspect="auto",
           interpolation="nearest")
plt.title("Image Created from the Given Array")
plt.xlabel("Column")
plt.yticks([])
plt.colorbar(label="Intensity")

# Save output image
plt.savefig(OUTPUT_DIR / "array_image.png",
            dpi=150, bbox_inches="tight")
plt.show()
plt.close()

print("Image dimensions:", image_1d.shape)

# --------------------------------------------------
# 1. INPUT ARRAY
# --------------------------------------------------

Ip = np.array([
    5, 5, 4, 3, 2, 1, 0, 0, 0, 6,
    0, 0, 0, 1, 3, 1, 0, 0, 0, 0,
    7, 7, 5
], dtype=float)

# Sample positions for the original array
x = np.arange(len(Ip))

# --------------------------------------------------
# 2. FIRST-ORDER DERIVATIVE
# D1[i] = Ip[i+1] - Ip[i]
# --------------------------------------------------

D1 = np.diff(Ip)
x1 = np.arange(1, len(Ip))

# --------------------------------------------------
# 3. SECOND-ORDER DERIVATIVE
# D2[i] = D1[i+1] - D1[i]
# --------------------------------------------------

D2 = np.diff(Ip, n=2)
x2 = np.arange(2, len(Ip))

# --------------------------------------------------
# 4. PRINT NUMERICAL RESULTS
# --------------------------------------------------

print("Original Input Array:")
print(Ip)

print("\nFirst-Order Derivative:")
print(D1)

print("\nSecond-Order Derivative:")
print(D2)

# --------------------------------------------------
# 5. PLOT ALL THREE GRAPHS
# --------------------------------------------------

fig, axes = plt.subplots(3, 1, figsize=(13, 10))

# Original array
axes[0].plot(x, Ip, 'o--')
axes[0].set_title("Original Input Array")
axes[0].set_xlabel("Sample Index")
axes[0].set_ylabel("Intensity")
axes[0].set_xticks(x)
axes[0].grid(True)

# First derivative
axes[1].plot(x1, D1, 'o--')
axes[1].axhline(0, linewidth=1)
axes[1].set_title("First-Order Derivative")
axes[1].set_xlabel("Sample Index")
axes[1].set_ylabel("First Derivative")
axes[1].set_xticks(x1)
axes[1].grid(True)

# Second derivative
axes[2].plot(x2, D2, 'o--')
axes[2].axhline(0, linewidth=1)
axes[2].set_title("Second-Order Derivative")
axes[2].set_xlabel("Sample Index")
axes[2].set_ylabel("Second Derivative")
axes[2].set_xticks(x2)
axes[2].grid(True)

plt.tight_layout()

# Save derivative graph
fig.savefig(OUTPUT_DIR / "array_derivatives.png",
            dpi=150, bbox_inches="tight")
plt.show()
plt.close(fig)

# ==========================================
# 1. LOAD THREE IMAGES USING DATASET PATHS
# ==========================================

V1 = cv2.imread(str(INPUT_DIR / "horizontal.png"),
                cv2.IMREAD_GRAYSCALE)
V2 = cv2.imread(str(INPUT_DIR / "vertical-line.jpg"),
                cv2.IMREAD_GRAYSCALE)
V3 = cv2.imread(str(INPUT_DIR / "diagonal.jpg"),
                cv2.IMREAD_GRAYSCALE)

if V1 is None:
    raise FileNotFoundError(f"Could not read: {INPUT_DIR / 'horizontal.png'}")
if V2 is None:
    raise FileNotFoundError(f"Could not read: {INPUT_DIR / 'vertical-line.jpg'}")
if V3 is None:
    raise FileNotFoundError(f"Could not read: {INPUT_DIR / 'diagonal.jpg'}")

# Resize to 25 x 25 if required
V1 = cv2.resize(V1, (25, 25))
V2 = cv2.resize(V2, (25, 25))
V3 = cv2.resize(V3, (25, 25))


# ==========================================
# 2. DEFINE FIVE KERNELS
# ==========================================

K1 = np.array([[1, 1, 1],
               [1,-8, 1],
               [1, 1, 1]], dtype=np.float32)

K2 = np.array([[-1,-1,-1],
               [ 2, 2, 2],
               [-1,-1,-1]], dtype=np.float32)

K3 = np.array([[-1, 2,-1],
               [-1, 2,-1],
               [-1, 2,-1]], dtype=np.float32)

K4 = np.array([[ 2,-1,-1],
               [-1, 2,-1],
               [-1,-1, 2]], dtype=np.float32)

K5 = np.array([[-1,-1, 2],
               [-1, 2,-1],
               [ 2,-1,-1]], dtype=np.float32)

kernels = [K1, K2, K3, K4, K5]
kernel_names = [
    "K1: Laplacian",
    "K2: Horizontal",
    "K3: Vertical",
    "K4: Main diagonal",
    "K5: Opposite diagonal"
]


# ==========================================
# 3. APPLY KERNELS TO AN IMAGE
# ==========================================

def apply_kernels(image):
    image = image.astype(np.float32)

    results = []

    for kernel in kernels:
        result = cv2.filter2D(
            image,
            cv2.CV_32F,
            kernel,
            borderType=cv2.BORDER_CONSTANT
        )
        results.append(result)

    return results

results_V1 = apply_kernels(V1)
results_V2 = apply_kernels(V2)
results_V3 = apply_kernels(V3)


# ==========================================
# 4. DISPLAY RESPONSE IMAGES
# ==========================================

def display_response(result):
    # Absolute value shows edge-response strength.
    response = np.abs(result)

    if response.max() > 0:
        response = response / response.max() * 255

    return response

def show_results(original, results, image_label, output_filename):

    fig, axes = plt.subplots(2, 3, figsize=(10, 7))
    axes = axes.ravel()

    axes[0].imshow(original, cmap="gray",
                   vmin=0, vmax=255,
                   interpolation="nearest")
    axes[0].set_title(f"{image_label}: Original")

    for i, result in enumerate(results):
        axes[i + 1].imshow(
            display_response(result),
            cmap="gray",
            vmin=0,
            vmax=255,
            interpolation="nearest"
        )
        axes[i + 1].set_title(kernel_names[i])
        axes[i + 1].axis("off")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()

    # Save the combined original + five kernel responses
    fig.savefig(OUTPUT_DIR / output_filename,
                dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)


# Results for all three images
show_results(V1, results_V1, "V1: Horizontal Line",
             "horizontal_output.png")
show_results(V2, results_V2, "V2: Vertical Line",
             "vertical_output.png")
show_results(V3, results_V3, "V3: Diagonal Line",
             "diagonal_output.png")


# ==========================================
# 5. PRINT NUMERICAL RESULTS
# ==========================================

for label, results in [
    ("V1: Horizontal Line", results_V1),
    ("V2: Vertical Line", results_V2),
    ("V3: Diagonal Line", results_V3)
]:
    print(f"\nResults for {label}")

    for i, result in enumerate(results):
        print(f"\nKernel K{i+1}")
        print("Minimum:", result.min())
        print("Maximum:", result.max())
        print(np.round(result, 2))

print("\nAll five output images saved in:", OUTPUT_DIR)
