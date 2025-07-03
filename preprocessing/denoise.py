import cv2

def apply_fast_denoising(frame):
    dst = cv2.fastNlMeansDenoisingColored(frame,None,11,6,7,21)
    return dst

def apply_gaussian_blur(frame):
    return cv2.GaussianBlur(frame,(5,5),0)

def apply_bilateral_filter(frame):
    return cv2.bilateralFilter(frame,9,75,75)

def apply_median_blur(frame):
    return cv2.medianBlur(frame,5)