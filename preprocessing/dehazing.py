import cv2
import numpy as np


# This function provides a more generalized approach to "dehaze" by enhancing contrast
# and reducing atmospheric light effects, which can mimic dehazing for snowy conditions.

def estimate_atmospheric_light(img: np.ndarray, patch_size: int = 15) -> np.ndarray:
    """
    Estimates atmospheric light from the brightest pixels in the darkest channel.
    A simplified approach for snowy scenes.
    """
    # Convert to float to avoid overflow in calculations
    img_float = img.astype(np.float32) / 255.0
    min_channel = np.min(img_float, axis=2) # Get dark channel

    # Find the top 0.1% brightest pixels in the dark channel
    flat_min_channel = min_channel.flatten()
    top_pixels_indices = np.argsort(flat_min_channel)[::-1][:int(len(flat_min_channel) * 0.001)]

    # Get corresponding pixel values from original image
    rows, cols = np.unravel_index(top_pixels_indices, min_channel.shape)
    atmospheric_light = np.mean(img_float[rows, cols], axis=0)

    # Ensure it's not all zeros, avoid division by zero later
    if np.all(atmospheric_light == 0):
        atmospheric_light = np.array([0.1, 0.1, 0.1]) # Small non-zero default

    return atmospheric_light

def apply_dehazing(frame: np.ndarray, t_min: float = 0.1, omega: float = 0.95, patch_size: int = 15) -> np.ndarray:
    """
    Applies a simplified dehazing technique suitable for snowy conditions.
    This method estimates atmospheric light and refines transmittance.
    t_min: Minimum transmission (prevents extreme enhancement in dense fog/snow).
    omega: Factor to control the amount of haze removed (0.0 to 1.0).
    """
    frame_float = frame.astype(np.float32) / 255.0

    # Estimate atmospheric light (A)
    atmospheric_light = estimate_atmospheric_light(frame, patch_size)

    # Calculate dark channel (J_dark)
    dark_channel = cv2.erode(np.min(frame_float, axis=2),
                             cv2.getStructuringElement(cv2.MORPH_RECT, (patch_size, patch_size)))

    # Estimate transmission map (t)
    # t(x) = 1 - omega * J_dark(x) / A_dark (simplified)
    # Using the average of atmospheric light for simplicity for division
    t = 1 - omega * dark_channel / np.max(atmospheric_light)
    t = np.maximum(t, t_min) # Ensure minimum transmission

    # Recover the scene radiance (J)
    # J(x) = (I(x) - A) / max(t(x), t_min) + A
    recovered_frame = np.zeros_like(frame_float)
    for c in range(3):
        recovered_frame[:, :, c] = (frame_float[:, :, c] - atmospheric_light[c]) / t + atmospheric_light[c]

    # Clip values to [0, 1] and convert back to 8-bit
    recovered_frame = np.clip(recovered_frame * 255, 0, 255).astype(np.uint8)

    return recovered_frame