import cv2
import numpy as np
import matplotlib.pyplot as plt

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    masked_img = cv2.bitwise_and(img, mask)
    return masked_img

def draw_lines(img, lines):
    line_img = np.zeros_like(img)
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(line_img, (x1, y1), (x2, y2), (255, 0, 0), 10)
    img = cv2.addWeighted(img, 0.8, line_img, 1, 0)
    return img

# Load the image
image = cv2.imread('22.jpg')
image_copy = np.copy(image)

# Convert to grayscale
gray = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur
blur = cv2.GaussianBlur(gray, (5, 3), 0)

# Canny edge detection
edges = cv2.Canny(blur, 50, 150)

# Define the region of interest centered and covering half the height
imshape = image.shape
bottom_left = (imshape[1] * 0.3, imshape[0])  # 10% from the left of the screen to the bottom
top_left = (imshape[1] * 0.3, imshape[0] * 0.4)  # 40% from the left, halfway up
top_right = (imshape[1] * 0.3, imshape[0] * 0.4)  # 60% from the left, halfway up
bottom_right = (imshape[1] * 0.9, imshape[0])  # 90% from the left to the bottom
vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)

masked_edges = region_of_interest(edges, vertices)

# Hough transform for line detection
lines = cv2.HoughLinesP(masked_edges, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=150)

# Draw the lines on the original image
lines_detected = draw_lines(image_copy, lines)

# Display the result
plt.figure(figsize=(10,10))
plt.imshow(cv2.cvtColor(lines_detected, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
