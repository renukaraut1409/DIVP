import cv2

# Read original color image
original = cv2.imread(
    "Dataset/Lab_2/reference.jpg"
)

# Convert original to grayscale
reference = cv2.cvtColor(
    original,
    cv2.COLOR_BGR2GRAY
)

# Create darker grayscale source
source1 = cv2.convertScaleAbs(
    reference,
    alpha=0.6,
    beta=0
)

# Create brighter grayscale source
source2 = cv2.convertScaleAbs(
    reference,
    alpha=1.3,
    beta=0
)

# Save all three grayscale images
cv2.imwrite(
    "Dataset/Lab_2/reference_gray.jpg",
    reference
)

cv2.imwrite(
    "Dataset/Lab_2/source1.jpg",
    source1
)

cv2.imwrite(
    "Dataset/Lab_2/source2.jpg",
    source2
)

print("Grayscale images created successfully!")