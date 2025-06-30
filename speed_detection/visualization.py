import cv2 as cv
from utils import draw_label_with_bg


def draw_ground_truth_speeds(frame, frame_count, gt_cars, font_scale=0.6):
    """
    Draws ground truth speed labels and circles for all cars visible in the current frame.

    Args:
        frame (ndarray): Current frame image.
        frame_count (int): Current frame number.
        gt_cars (List[dict]): Ground truth cars list with 'frames', 'posX', 'posY', 'real_speed'.
        font_scale (float): Scale for the speed label font.
    """
    for car in gt_cars:
        if frame_count in car["frames"]:
            i = car["frames"].index(frame_count)
            orig_x, orig_y = int(car["posX"][i]), int(car["posY"][i])
            speed = car.get("real_speed")
            if speed is not None:
                draw_label_with_bg(
                    frame,
                    f"GT: {speed:.1f} km/h",
                    (orig_x, orig_y - 25),
                    font_scale=font_scale,
                    bg_color=(0, 100, 255),
                )
                cv.circle(frame, (orig_x, orig_y), 5, (0, 100, 255), -1)


def draw_rois(frame, roi_data):
    overlay = frame.copy()
    for roi in roi_data:
        if roi["completed"]:
            cv.fillPoly(overlay, [roi["roi_pts"]], roi["color"])
            top_mid = tuple(((roi["roi_pts"][0] + roi["roi_pts"][1]) // 2))
            bottom_mid = tuple(((roi["roi_pts"][2] + roi["roi_pts"][3]) // 2))
            cv.line(frame, top_mid, bottom_mid, (0, 255, 0), 2)
    return cv.addWeighted(overlay, 0.3, frame, 0.7, 0)


def draw_tracks(frame, tracks, roi_data, vehicle_data):
    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        l, t, r, b = map(int, track.to_ltrb())

        for roi in roi_data:
            if roi["completed"] and track_id in vehicle_data[roi["id"]]:
                track_data = vehicle_data[roi["id"]][track_id]

                if (
                    track_data["kalman_speed"] is not None
                    and track_data["kalman_speed"] > 0
                ):
                    speed_text = (
                        f"Speed: {track_data['kalman_speed']:.1f} km/h"
                    )
                    draw_label_with_bg(
                        frame,
                        speed_text,
                        (l, t - 26),
                        font_scale=0.6,
                        bg_color=(140, 0, 0),
                    )

                # Draw motion trail
                for i in range(1, len(track_data["position_history"])):
                    cv.line(
                        frame,
                        track_data["position_history"][i - 1],
                        track_data["position_history"][i],
                        roi["color"],
                        2,
                    )
