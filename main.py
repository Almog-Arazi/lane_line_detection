import numpy as np
import cv2

def process(image):
    image_g = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    threshold_low = 50
    threshold_high = 200
    image_canny = cv2.Canny(image_g, threshold_low, threshold_high)

    vertices = np.array([[(335,image.shape[0]), (870,image.shape[0]), (685,530), (550, 530)]], dtype=np.int32)
    cropped_image = region_of_interest(image_canny, vertices)
    rho = 2
    theta = np.pi/180
    threshold = 50
    min_line_len = 35
    max_line_gap = 30
    lines = cv2.HoughLinesP(cropped_image, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_image = draw_the_lines(image, lines, vertices)  # Updated to pass vertices
    return line_image

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def draw_the_lines(img, lines, vertices):
    left_lines = []  # Lines on the left side
    right_lines = []  # Lines on the right side
    for line in lines:
        for x1, y1, x2, y2 in line:
            # Calculate the slope; avoid division by zero
            if (x2 - x1) == 0:  # This would be a vertical line
                continue  # Proceed with vertical lines, since we're ignoring horizontal ones
            slope = (y2 - y1) / (x2 - x1)
            # Filter out horizontal lines based on slope threshold
            if abs(slope) < 0.5:  # Adjust this threshold to ignore lines that are too horizontal
                continue  # This excludes nearly horizontal lines
            if slope < 0:
                left_lines.append((slope, y1 - slope * x1))
            else:
                right_lines.append((slope, y1 - slope * x1))

    if left_lines:
        left_avg = np.average(left_lines, axis=0)
        x1, y1, x2, y2 = calculate_coordinates(img.shape, left_avg, vertices)
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 10)

    if right_lines:
        right_avg = np.average(right_lines, axis=0)
        x1, y1, x2, y2 = calculate_coordinates(img.shape, right_avg, vertices)
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 10)

    return img


def calculate_coordinates(shape, line_parameters, vertices):
    slope, intercept = line_parameters
    y1 = max(vertices[0][0][1], vertices[0][1][1])  # The higher (y-value) of the bottom vertices
    y2 = min(vertices[0][2][1], vertices[0][3][1])  # The lower (y-value) of the top vertices
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return x1, y1, x2, y2

cap = cv2.VideoCapture('./3.mp4')
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
size = (frame_width, frame_height)

result = cv2.VideoWriter('./result.mp4', 
                         cv2.VideoWriter_fourcc(*'mp4v'), 
                         20, size)

while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        frame_with_roi = process(frame)
        result.write(frame_with_roi)
    else:
        break

cap.release()
result.release()
cv2.destroyAllWindows()
