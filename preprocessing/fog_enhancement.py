import cv2

def remove_fog_contrast(frame):
    # Convert to LAB color space
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)

    # Split channels
    l, a, b = cv2.split(lab)

    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)

    # Merge channels
    limg = cv2.merge((cl, a, b))

    # Convert back to BGR
    less_fog_frame = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    return less_fog_frame
