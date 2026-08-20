import cv2
import numpy as np

# Read the reference image
reference = cv2.imread(
    "Dataset/Lab_2/reference.jpg",
    cv2.IMREAD_GRAYSCALE
)

# Read the two source images
source1 = cv2.imread(
    "Dataset/Lab_2/source1.jpg",
    cv2.IMREAD_GRAYSCALE
)

source2 = cv2.imread(
    "Dataset/Lab_2/source2.jpg",
    cv2.IMREAD_GRAYSCALE
)

print("reference shape:", reference.shape)
print("Source 1 shape:", source1.shape)
print("Source 2 shape:", source2.shape)

print("Reference data type:", reference.dtype)
print("Source 1 data type:", source1.dtype)

def calculate_histogram(image):

    histogram = np.zeros(256, dtype=np.int64)

    for pixel in image.flatten():
        histogram[pixel] += 1

    return histogram

reference_hist = calculate_histogram(reference)
source1_hist = calculate_histogram(source1)
source2_hist = calculate_histogram(source2)

def calculate_pdf(histogram):

    total_pixels = histogram.sum()

    pdf = histogram / total_pixels

    return pdf

reference_pdf = calculate_pdf(reference_hist)
source1_pdf = calculate_pdf(source1_hist)
source2_pdf = calculate_pdf(source2_hist)

print("Reference PDF sum:", reference_pdf.sum())
print("Source 1 PDF sum:", source1_pdf.sum())
print("Source 2 PDF sum:", source2_pdf.sum())

def calculate_cdf(pdf):

    cdf = np.cumsum(pdf)

    return cdf

reference_cdf = calculate_cdf(reference_pdf)
source1_cdf = calculate_cdf(source1_pdf)
source2_cdf = calculate_cdf(source2_pdf)

print("Reference CDF last value:", reference_cdf[-1])
print("Source 1 CDF last value:", source1_cdf[-1])
print("Source 2 CDF last value:", source2_cdf[-1])

def create_mapping(source_cdf, reference_cdf):

    mapping = np.zeros(256, dtype=np.uint8)

    for source_intensity in range(256):

        # CDF value of the current source intensity
        source_value = source_cdf[source_intensity]

        # Difference between source CDF and every reference CDF value
        difference = np.abs(reference_cdf - source_value)

        # Find the reference intensity with minimum difference
        reference_intensity = np.argmin(difference)

        # Store the mapping
        mapping[source_intensity] = reference_intensity

    return mapping

mapping1 = create_mapping(
    source1_cdf,
    reference_cdf
)

mapping2 = create_mapping(
    source2_cdf,
    reference_cdf
)
matched_source1 = mapping1[source1]
matched_source2 = mapping2[source2]
cv2.imwrite(
    "Lab_2/Outputs/matched_source1.jpg",
    matched_source1
)

cv2.imwrite(
    "Lab_2/Outputs/matched_source2.jpg",
    matched_source2
)
matched1_hist = calculate_histogram(matched_source1)
matched2_hist = calculate_histogram(matched_source2)

matched1_pdf = calculate_pdf(matched1_hist)
matched2_pdf = calculate_pdf(matched2_hist)

matched1_cdf = calculate_cdf(matched1_pdf)
matched2_cdf = calculate_cdf(matched2_pdf)



import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))

plt.plot(reference_hist, label="Reference")
plt.plot(source1_hist, label="Source 1")
plt.plot(source2_hist, label="Source 2")

plt.xlabel("Intensity")
plt.ylabel("Number of Pixels")
plt.title("Histograms of Images")
plt.legend()

plt.show()

plt.figure(figsize=(10, 5))

plt.plot(reference_cdf, label="Reference")
plt.plot(source1_cdf, label="Source 1")
plt.plot(source2_cdf, label="Source 2")

plt.xlabel("Intensity")
plt.ylabel("CDF")
plt.title("CDF of Images")
plt.legend()

plt.show()

######## histogram

plt.figure(figsize=(10, 5))

plt.plot(reference_hist, label="Reference")
plt.plot(matched1_hist, label="Matched Source 1")
plt.plot(matched2_hist, label="Matched Source 2")

plt.xlabel("Intensity")
plt.ylabel("Number of Pixels")
plt.title("Histogram After Matching")
plt.legend()

plt.show()

#CDF

plt.figure(figsize=(10, 5))

plt.plot(reference_cdf, label="Reference")
plt.plot(matched1_cdf, label="Matched Source 1")
plt.plot(matched2_cdf, label="Matched Source 2")

plt.xlabel("Intensity")
plt.ylabel("CDF")
plt.title("CDF After Histogram Matching")
plt.legend()

plt.show()

def create_analytical_target():

    intensity = np.arange(256)

    # Create a Gaussian-shaped distribution
    target_pdf = np.exp(
        -((intensity - 80) ** 2) / (2 * 35 ** 2)
    )

    # Normalize so total probability = 1
    target_pdf = target_pdf / target_pdf.sum()

    return target_pdf
target_pdf = create_analytical_target()

target_cdf = np.cumsum(target_pdf)
analytical_mapping = create_mapping(
    source1_cdf,
    target_cdf
)

analytical_result = analytical_mapping[source1]

cv2.imwrite(
    "Lab_2/Outputs/analytical_target_result.jpg",
    analytical_result
)
analytical_hist = calculate_histogram(analytical_result)

analytical_pdf = calculate_pdf(
    analytical_hist
)

analytical_cdf = calculate_cdf(
    analytical_pdf
)

plt.figure(figsize=(10, 5))

plt.plot(target_pdf)

plt.xlabel("Intensity")
plt.ylabel("Probability")
plt.title("Analytical Target Histogram")

plt.show()
plt.figure(figsize=(10, 5))

plt.plot(target_cdf)

plt.xlabel("Intensity")
plt.ylabel("CDF")
plt.title("Analytical Target CDF")

plt.show()

plt.figure(figsize=(10, 5))

plt.plot(target_cdf, label="Analytical Target")
plt.plot(analytical_cdf, label="Result")

plt.xlabel("Intensity")
plt.ylabel("CDF")
plt.title("Analytical Target vs Result")
plt.legend()

plt.show()