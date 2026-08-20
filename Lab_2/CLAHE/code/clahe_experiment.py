import cv2
import matplotlib.pyplot as plt
import os


# ============================================================
# 1. READ INPUT IMAGE
# ============================================================

# We use the grayscale image throughout this lab.
image = cv2.imread(
    "Dataset/Lab_2/reference_gray.jpg",
    cv2.IMREAD_GRAYSCALE
)

if image is None:
    raise FileNotFoundError("Could not load reference_gray.jpg")

print("Image shape:", image.shape)
print("Image dtype:", image.dtype)


# ============================================================
# 2. CREATE OUTPUT FOLDER
# ============================================================

os.makedirs("Lab_2/CLAHE/Outputs", exist_ok=True)


# ============================================================
# 3. DISPLAY ORIGINAL IMAGE
# ============================================================

plt.figure(figsize=(8, 5))

plt.imshow(image, cmap="gray")
plt.title("Original Grayscale Image")
plt.axis("off")

plt.tight_layout()
plt.show()


# ============================================================
# 4. GLOBAL HISTOGRAM EQUALIZATION
# ============================================================

# Global histogram equalization considers the ENTIRE image
# as one distribution and creates ONE transformation.
#
# This is our baseline for comparison with CLAHE.

equalized = cv2.equalizeHist(image)


# Save global equalization result
cv2.imwrite(
    "Lab_2/CLAHE/Outputs/global_equalization.jpg",
    equalized
)


# ============================================================
# 5. DIVIDE IMAGE CONCEPTUALLY INTO TILES
# ============================================================

# CLAHE works locally instead of using one histogram
# for the entire image.
#
# Here we use an 8 x 8 grid = 64 tiles.

tile_rows = 8
tile_cols = 8

height, width = image.shape

tile_height = height // tile_rows
tile_width = width // tile_cols

print("Image size:", height, "x", width)
print("Tile size:", tile_height, "x", tile_width)


# Display the tile boundaries
plt.figure(figsize=(10, 6))

plt.imshow(image, cmap="gray")

# Draw vertical tile boundaries
for col in range(1, tile_cols):
    x = col * tile_width
    plt.axvline(x, linewidth=0.8)

# Draw horizontal tile boundaries
for row in range(1, tile_rows):
    y = row * tile_height
    plt.axhline(y, linewidth=0.8)

plt.title("Image Divided into CLAHE Tiles")
plt.axis("off")

plt.tight_layout()
plt.show()


# ============================================================
# 6. CLAHE PRACTICAL IMPLEMENTATION
# ============================================================

# clipLimit controls how much contrast enhancement is allowed.
#
# A larger clipLimit:
#     → stronger contrast enhancement
#     → but potentially more noise amplification
#
# A smaller clipLimit:
#     → less enhancement
#     → less chance of noise amplification

clip_limit = 2.0

# tileGridSize determines the number of local regions.
#
# (8, 8) means the image is divided into an 8 x 8 grid.

tile_grid_size = (8, 8)


# Create the CLAHE object
clahe = cv2.createCLAHE(
    clipLimit=clip_limit,
    tileGridSize=tile_grid_size
)


# Apply CLAHE to the grayscale image
clahe_result = clahe.apply(image)


# Save CLAHE result
cv2.imwrite(
    "Lab_2/CLAHE/Outputs/clahe_result.jpg",
    clahe_result
)


# ============================================================
# 7. COMPARE ORIGINAL, GLOBAL HE AND CLAHE
# ============================================================

plt.figure(figsize=(15, 5))


# Original
plt.subplot(1, 3, 1)

plt.imshow(image, cmap="gray")
plt.title("Original")
plt.axis("off")


# Global Histogram Equalization
plt.subplot(1, 3, 2)

plt.imshow(equalized, cmap="gray")
plt.title("Global Histogram Equalization")
plt.axis("off")


# CLAHE
plt.subplot(1, 3, 3)

plt.imshow(clahe_result, cmap="gray")
plt.title("CLAHE")
plt.axis("off")


plt.tight_layout()
plt.show()


# ============================================================
# 8. COMPARE HISTOGRAMS
# ============================================================

plt.figure(figsize=(12, 5))


plt.subplot(1, 3, 1)

plt.hist(image.ravel(), bins=256, range=(0, 256))
plt.title("Original Histogram")
plt.xlabel("Intensity")
plt.ylabel("Pixels")


plt.subplot(1, 3, 2)

plt.hist(equalized.ravel(), bins=256, range=(0, 256))
plt.title("Global HE Histogram")
plt.xlabel("Intensity")
plt.ylabel("Pixels")


plt.subplot(1, 3, 3)

plt.hist(clahe_result.ravel(), bins=256, range=(0, 256))
plt.title("CLAHE Histogram")
plt.xlabel("Intensity")
plt.ylabel("Pixels")


plt.tight_layout()
plt.show()


# ============================================================
# 9. PARAMETER SWEEP — CLIP LIMIT
# ============================================================

# The lab asks us to investigate how clip limit affects
# the result.
#
# We test several values while keeping the tile size fixed.

clip_limits = [1.0, 2.0, 4.0, 8.0]

plt.figure(figsize=(12, 8))

for i, clip in enumerate(clip_limits):

    clahe_test = cv2.createCLAHE(
        clipLimit=clip,
        tileGridSize=(8, 8)
    )

    result = clahe_test.apply(image)

    plt.subplot(2, 2, i + 1)

    plt.imshow(result, cmap="gray")
    plt.title(f"Clip Limit = {clip}")
    plt.axis("off")

    # Save each result
    cv2.imwrite(
        f"Lab_2/CLAHE/Outputs/clip_{clip}.jpg",
        result
    )

plt.tight_layout()
plt.show()


# ============================================================
# 10. PARAMETER SWEEP — TILE SIZE
# ============================================================

# Now we keep clipLimit fixed and change the tile grid.
#
# Smaller tiles:
#     → more local enhancement
#     → potentially more noise / artifacts
#
# Larger tiles:
#     → behavior becomes more global
#     → less local adaptation

tile_sizes = [
    (4, 4),
    (8, 8),
    (16, 16)
]

plt.figure(figsize=(12, 5))

for i, tile_size in enumerate(tile_sizes):

    clahe_test = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=tile_size
    )

    result = clahe_test.apply(image)

    plt.subplot(1, 3, i + 1)

    plt.imshow(result, cmap="gray")
    plt.title(f"Tile Size = {tile_size}")
    plt.axis("off")

    # Save each result
    cv2.imwrite(
        f"Lab_2/CLAHE/Outputs/tile_{tile_size[0]}x{tile_size[1]}.jpg",
        result
    )

plt.tight_layout()
plt.show()


print("\nCLAHE experiment completed successfully!")
print("Outputs saved in Lab_2/CLAHE/Outputs/")