import matplotlib.pylab as plt
import cv2
import numpy as np


def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    match_mask_color = 255
    cv2.fillPoly(mask, vertices, match_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def draw_the_lines(img, lines):
    img = np.copy(img)
    blank_image = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    if lines is not None :
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(blank_image, (x1,y1), (x2,y2), (0, 255, 0), thickness=5)

        img = cv2.addWeighted(img, 0.8, blank_image, 1, 0.0)
    return img

def filter_lines_by_slope(lines, min_slope=0.3, max_slope=3.0):
    """Filter lines based on slope to remove horizontal/nearly horizontal lines (likely from cars)"""
    filtered_lines = []
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                if x2 - x1 == 0:  # Avoid division by zero
                    continue
                slope = abs((y2 - y1) / (x2 - x1))
                if min_slope <= slope <= max_slope:
                    filtered_lines.append([[x1, y1, x2, y2]])
    return np.array(filtered_lines) if filtered_lines else None

def filter_lines_by_length(lines, min_length=50):
    """Filter out short lines that might be from car edges"""
    filtered_lines = []
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                if length >= min_length:
                    filtered_lines.append([[x1, y1, x2, y2]])
    return np.array(filtered_lines) if filtered_lines else None

def process_with_color_filtering(image):
    """Alternative approach using color filtering to focus on lane markings"""
    print(image.shape)
    height = image.shape[0]
    width = image.shape[1]


    # Convert to HSV for better color filtering
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # Define range for white and yellow lane markings
    # White lane markings
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 30, 255])
    white_mask = cv2.inRange(hsv, lower_white, upper_white)

    # Yellow lane markings
    lower_yellow = np.array([15, 100, 100])
    upper_yellow = np.array([35, 255, 255])
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # Combine masks
    lane_mask = cv2.bitwise_or(white_mask, yellow_mask)

    # Apply Gaussian blur
    blurred_mask = cv2.GaussianBlur(lane_mask, (5, 5), 0)

    # ROI
    region_of_interest_vertices = [
        (0, height),
        (width/2, height/2),
        (width, height)
    ]

    cropped_image = region_of_interest(blurred_mask,
                                       np.array([region_of_interest_vertices], np.int32))

    # Edge detection on the filtered image
    canny_image = cv2.Canny(cropped_image, 50, 150)

    lines = cv2.HoughLinesP(canny_image,
                            rho=1,
                            theta=np.pi/180,
                            threshold=30,
                            lines=np.array([]),
                            minLineLength=80,
                            maxLineGap=40)

    # Filter lines
    lines = filter_lines_by_slope(lines, min_slope=0.3, max_slope=3.0)
    lines = filter_lines_by_length(lines, min_length=60)

    image_with_lines = draw_the_lines(image, lines)
    return image_with_lines
def process_gemini(image):
    print(image.shape)
    height = image.shape[0]
    width = image.shape[1]
    fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=80, detectShadows=True)
    fgmask = fgbg.apply(image)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel=np.ones((3,3),np.uint8))
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel=np.ones((3,3),np.uint8))
    background_mask = cv2.bitwise_not(fgmask)
    static_image = cv2.bitwise_and(image, image, mask=background_mask)
    region_of_interest_vertices = [
        (0, height),
        (width/2, height/2),
        (width, height)
    ]
    gray_image = cv2.cvtColor(static_image, cv2.COLOR_RGB2GRAY)
    canny_image = cv2.Canny(gray_image, 150,200)
    cropped_image = region_of_interest(canny_image,
                                       np.array([region_of_interest_vertices], np.int32),)
    lines = cv2.HoughLinesP(cropped_image,
                            rho=2,
                            theta=np.pi/180,
                            threshold=130,
                            lines=np.array([]),
                            minLineLength=80,
                            maxLineGap=100)
    lines = filter_lines_by_slope(lines,min_slope = 0.3, max_slope=3.0)
    lines = filter_lines_by_length(lines,min_length=80)

    image_with_lines = draw_the_lines(image, lines)
    cv2.imshow("cropped_image",cropped_image)
    cv2.imshow("static img",static_image)
    return image_with_lines

def process_frame(image):
    print(image.shape)
    height = image.shape[0]
    width = image.shape[1]
    region_of_interest_vertices = [
        (0, height),
        (width/2, height/2),
        (width, height)
    ]
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray_image, (5, 5), 1.4)
    canny_image = cv2.Canny(blurred, 150,200)
    cropped_image = region_of_interest(canny_image,
                                       np.array([region_of_interest_vertices], np.int32),)
    lines = cv2.HoughLinesP(cropped_image,
                            rho=2,
                            theta=np.pi/180,
                            threshold=20,
                            lines=np.array([]),
                            minLineLength=40,
                            maxLineGap=100)
    lines = filter_lines_by_slope(lines,min_slope = 0.3, max_slope=3.0)
    lines = filter_lines_by_length(lines,min_length=80)

    image_with_lines = draw_the_lines(image, lines)
    cv2.imshow("cropped_image",cropped_image)
    cv2.imshow("blurred",blurred)
    return image_with_lines
cap = cv2.VideoCapture('test.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    frame = process_gemini(frame)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()