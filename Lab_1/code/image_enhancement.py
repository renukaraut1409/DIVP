import os
import math
from PIL import Image


# ============================================================
# DVP LAB 1
# IMAGE ENHANCEMENT - TRANSFORMATION FUNCTIONS
#
# 1. Negative       S = L - 1 - r
# 2. Gamma          S = c(r^gamma)
# 3. Log            S = c log(1 + r)
#
# Two implementations:
# 1. Manual - direct pixel-by-pixel processing
# 2. Tool/Helper - reusable Image class and map_pixels()
# ============================================================


# ============================================================
# 1. PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

# CHANGE THIS TO YOUR ACTUAL JPG FILE NAME
INPUT_PATH = os.path.join(
    BASE_DIR,
    "..",
    "Dataset",
    "Lab_1",
    "reference.jpg"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "Image Output"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 2. LOAD COLOR IMAGE
# ============================================================

img = Image.open(INPUT_PATH)

print("=" * 55)
print("DVP LAB 1 - IMAGE ENHANCEMENT")
print("=" * 55)

print("\nInput image:", INPUT_PATH)
print("Original mode:", img.mode)
print("Original size:", img.size)


# ============================================================
# 3. CONVERT COLOR IMAGE TO GRAYSCALE
# ============================================================

img = img.convert("L")

width, height = img.size

print("Converted to grayscale.")
print("Image size:", width, "x", height)


# Convert image into list of intensity values
pixels = list(img.getdata())


# ============================================================
# 4. HELPER FUNCTION
# ============================================================

def clamp(value):
    """Keep intensity value between 0 and 255."""

    if value < 0:
        return 0

    if value > 255:
        return 255

    return int(value)


# ============================================================
# 5. MANUAL IMPLEMENTATION
# ============================================================


# ------------------------------------------------------------
# NEGATIVE TRANSFORMATION
#
# Formula:
# S = L - 1 - r
#
# For 8-bit image:
# S = 255 - r
# ------------------------------------------------------------

def negative_manual(pixels):

    result = []

    L = 256

    for r in pixels:

        s = L - 1 - r

        result.append(
            clamp(s)
        )

    return result


# ------------------------------------------------------------
# GAMMA TRANSFORMATION
#
# Formula:
# S = c(r^gamma)
#
# r is normalized to [0,1]
#
# gamma = 0.5
# ------------------------------------------------------------

def gamma_manual(pixels, gamma, c=1.0):

    result = []

    for r in pixels:

        normalized_r = r / 255.0

        s = c * (
            normalized_r ** gamma
        )

        s = s * 255

        result.append(
            clamp(s)
        )

    return result


# ------------------------------------------------------------
# LOG TRANSFORMATION
#
# Formula:
# S = c log(1 + r)
#
# For an 8-bit image:
# c = 255 / log(256)
# ------------------------------------------------------------

def log_manual(pixels):

    result = []

    c = 255 / math.log(256)

    for r in pixels:

        s = c * math.log(1 + r)

        result.append(
            clamp(s)
        )

    return result


# ============================================================
# 6. REUSABLE IMAGE HELPER
# ============================================================

class ImageHelper:

    def __init__(self, width, height, pixels):

        self.width = width
        self.height = height
        self.pixels = pixels

    def map_pixels(self, function):

        result = []

        for pixel in self.pixels:

            result.append(
                function(pixel)
            )

        return ImageHelper(
            self.width,
            self.height,
            result
        )

    def save(self, filename):

        output = Image.new(
            "L",
            (self.width, self.height)
        )

        output.putdata(self.pixels)

        output.save(
            filename,
            quality=95
        )


# ============================================================
# 7. TOOL / HELPER IMPLEMENTATIONS
# ============================================================


def negative_tool(image):

    def transform(r):

        return 255 - r

    return image.map_pixels(transform)


def gamma_tool(image, gamma, c=1.0):

    def transform(r):

        normalized_r = r / 255.0

        s = c * (
            normalized_r ** gamma
        )

        return clamp(
            s * 255
        )

    return image.map_pixels(transform)


def log_tool(image):

    c = 255 / math.log(256)

    def transform(r):

        s = c * math.log(1 + r)

        return clamp(s)

    return image.map_pixels(transform)


# ============================================================
# 8. CREATE IMAGE HELPER OBJECT
# ============================================================

image_helper = ImageHelper(
    width,
    height,
    pixels
)


# ============================================================
# 9. SAVE ORIGINAL
# ============================================================

original_output = os.path.join(
    OUTPUT_DIR,
    "original_grayscale.jpg"
)

img.save(
    original_output,
    quality=95
)


# ============================================================
# 10. MANUAL TRANSFORMATIONS
# ============================================================

print("\nRunning manual transformations...")

negative_manual_result = negative_manual(
    pixels
)

gamma_manual_result = gamma_manual(
    pixels,
    gamma=0.5
)

log_manual_result = log_manual(
    pixels
)


# Save manual results

Image.new(
    "L",
    (width, height)
).save if False else None


def save_pixels(pixels, filename):

    output = Image.new(
        "L",
        (width, height)
    )

    output.putdata(pixels)

    output.save(
        os.path.join(
            OUTPUT_DIR,
            filename
        ),
        quality=95
    )


save_pixels(
    negative_manual_result,
    "negative_manual.jpg"
)

save_pixels(
    gamma_manual_result,
    "gamma_manual.jpg"
)

save_pixels(
    log_manual_result,
    "log_manual.jpg"
)


# ============================================================
# 11. TOOL / HELPER TRANSFORMATIONS
# ============================================================

print("Running tool/helper transformations...")

negative_tool_result = negative_tool(
    image_helper
)

gamma_tool_result = gamma_tool(
    image_helper,
    gamma=0.5
)

log_tool_result = log_tool(
    image_helper
)


# Save tool results

negative_tool_result.save(
    os.path.join(
        OUTPUT_DIR,
        "negative_tool.jpg"
    )
)

gamma_tool_result.save(
    os.path.join(
        OUTPUT_DIR,
        "gamma_tool.jpg"
    )
)

log_tool_result.save(
    os.path.join(
        OUTPUT_DIR,
        "log_tool.jpg"
    )
)


# ============================================================
# 12. CHECK OUTPUTS
# ============================================================

print("\n" + "=" * 55)
print("OUTPUT FILES")
print("=" * 55)

output_files = [
    "original_grayscale.jpg",
    "negative_manual.jpg",
    "gamma_manual.jpg",
    "log_manual.jpg",
    "negative_tool.jpg",
    "gamma_tool.jpg",
    "log_tool.jpg"
]

for filename in output_files:

    path = os.path.join(
        OUTPUT_DIR,
        filename
    )

    if os.path.exists(path):

        size = os.path.getsize(path)

        print(
            "✓",
            filename,
            f"({size / 1024:.2f} KB)"
        )

    else:

        print(
            "✗",
            filename,
            "NOT FOUND"
        )


# ============================================================
# 13. FINAL MESSAGE
# ============================================================

print("\n" + "=" * 55)
print("LAB 1 COMPLETED SUCCESSFULLY")
print("=" * 55)

print("\nTransformations:")
print("✓ Negative")
print("✓ Gamma correction (gamma = 0.5)")
print("✓ Log transformation")

print("\nImplementations:")
print("✓ Manual pixel-by-pixel implementation")
print("✓ Reusable Image/map_pixels implementation")

print("\nAll results saved in:")
print(OUTPUT_DIR)