import cv2
import numpy as np
import matplotlib.pyplot as plt


# -----------------------------
# Create source image
# -----------------------------

# Smooth horizontal grayscale gradient
source = np.tile(
    np.arange(256, dtype=np.uint8),
    (300, 1)
)


# -----------------------------
# Create reference image
# -----------------------------

reference = np.zeros((300, 256), dtype=np.uint8)

# Left half = dark
reference[:, :128] = 30

# Right half = bright
reference[:, 128:] = 220


# Save images
cv2.imwrite(
    "Dataset/Lab_2/failure_source.jpg",
    source
)

cv2.imwrite(
    "Dataset/Lab_2/failure_reference.jpg",
    reference
)


# Display them
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.imshow(source, cmap="gray")
plt.title("Failure Source")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(reference, cmap="gray")
plt.title("Failure Reference")
plt.axis("off")

plt.tight_layout()
plt.show()

def calculate_histogram(image):
    histogram = np.zeros(256, dtype=np.int64)

    for pixel in image.flatten():
        histogram[pixel] += 1

    return histogram


def calculate_cdf(histogram):
    pdf = histogram / histogram.sum()
    return np.cumsum(pdf)


def create_mapping(source_cdf, reference_cdf):

    mapping = np.zeros(256, dtype=np.uint8)

    for source_intensity in range(256):

        source_value = source_cdf[source_intensity]

        difference = np.abs(
            reference_cdf - source_value
        )

        reference_intensity = np.argmin(difference)

        mapping[source_intensity] = reference_intensity

    return mapping

source_hist = calculate_histogram(source)
reference_hist = calculate_histogram(reference)

source_cdf = calculate_cdf(source_hist)
reference_cdf = calculate_cdf(reference_hist)

mapping = create_mapping(
    source_cdf,
    reference_cdf
)

matched = mapping[source]

cv2.imwrite(
    "Lab_2/Outputs/failure_case_result.jpg",
    matched
)

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(source, cmap="gray")
plt.title("Source")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(reference, cmap="gray")
plt.title("Reference")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(matched, cmap="gray")
plt.title("Matched Result")
plt.axis("off")

plt.tight_layout()
plt.show()