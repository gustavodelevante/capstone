import numpy as np
import cv2
from matplotlib import pyplot as plt
import os


def grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def gaussian_blur(image, kernel_size):
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

def sobel_filters(image):
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1)
    return sobel_x, sobel_y

def gradient_magnitude_direction(sobel_x, sobel_y):
    magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    direction = np.arctan2(sobel_y, sobel_x)
    return magnitude, direction

def non_max_suppression(magnitude, direction):
    rows, cols = magnitude.shape
    result = np.zeros((rows, cols), dtype=np.float32)
    angle = direction * 180. / np.pi
    angle[angle < 0] += 180

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            q = 255
            r = 255

            if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                q = magnitude[i, j + 1]
                r = magnitude[i, j - 1]
            elif (22.5 <= angle[i, j] < 67.5):
                q = magnitude[i + 1, j - 1]
                r = magnitude[i - 1, j + 1]
            elif (67.5 <= angle[i, j] < 112.5):
                q = magnitude[i + 1, j]
                r = magnitude[i - 1, j]
            elif (112.5 <= angle[i, j] < 157.5):
                q = magnitude[i - 1, j - 1]
                r = magnitude[i + 1, j + 1]

            if (magnitude[i, j] >= q) and (magnitude[i, j] >= r):
                result[i, j] = magnitude[i, j]
            else:
                result[i, j] = 0

    return result

def hysteresis_thresholding(image, low_threshold, high_threshold):
    strong_edges = (image > high_threshold)
    weak_edges = (image >= low_threshold) & (image <= high_threshold)

    # Connectivity check
    for i in range(1, image.shape[0] - 1):
        for j in range(1, image.shape[1] - 1):
            if weak_edges[i, j]:
                if (strong_edges[i - 1:i + 2, j - 1:j + 2]).any():
                    strong_edges[i, j] = True
                else:
                    weak_edges[i, j] = False

    return strong_edges.astype(np.uint8) * 255

def canny_edge_detection(image, low_threshold, high_threshold, kernel_size=5):
    # Step 1: Convert image to grayscale
    gray = grayscale(image)

    # Step 2: Apply Gaussian blur
    blur = gaussian_blur(gray, kernel_size)

    # Step 3: Compute gradients
    sobel_x, sobel_y = sobel_filters(blur)

    # Step 4: Compute magnitude and direction of gradients
    magnitude, direction = gradient_magnitude_direction(sobel_x, sobel_y)

    # Step 5: Non-maximum suppression
    suppressed = non_max_suppression(magnitude, direction)

    # Step 6: Hysteresis thresholding
    edges = hysteresis_thresholding(suppressed, low_threshold, high_threshold)

    return edges


#image = cv2.imread(r"./mosaic3_.jpg")


#edges = canny_edge_detection(image, low_threshold=20, high_threshold=25, kernel_size=5)


#plt.subplot(121),plt.imshow(image,cmap = 'gray')
#plt.title('Original Image'), plt.xticks([]), plt.yticks([])
#plt.subplot(122),plt.imshow(edges,cmap = 'gray')
#plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

#plt.show()



# Assuming all necessary functions defined above are present here.

def process_images_from_folder(source_folder, destination_folder, low_threshold=20, high_threshold=25, kernel_size=5):
    # Ensure destination folder exists
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # List all jpg files in the source folder
    for filename in os.listdir(source_folder):
        if (filename.endswith(".jpg") or filename.endswith(".JPG")):  # Assuming we're working with jpg images
            # Construct full file path
            file_path = os.path.join(source_folder, filename)
            # Read the image
            image = cv2.imread(file_path)
            # Apply the canny edge detection
            edges = canny_edge_detection(image, low_threshold, high_threshold, kernel_size)
            # Construct the output file path
            output_file_path = os.path.join(destination_folder, filename)
            # Save the transformed image
            cv2.imwrite(output_file_path, edges)
            print(f"Processed and saved: {output_file_path}")


process_images_from_folder(r"C:\Users\gusta\OneDrive\Escritorio\Capstone\Crops Data\Cashew_only\Invalid" , r"C:\Users\gusta\OneDrive\Escritorio\Capstone\transformed_image\Invalid")
