import cv2 as cv
import numpy as np
from config import (
    FONT,
    COL_RED,
    COL_GRN,
    COL_BLU,
    COL_GREY,
    COL_HELPER,
    ROI_COLORS,
    LANE_LEN_M,
    GAP_LEN_M,
    LANE_WIDTH_M,
)


class ROISelector:
    def __init__(self, first_frame, n_rois):
        self.first_frame = first_frame
        self.n_rois = n_rois
        self.roi_data = []
        self.current_roi = 0
        self.cursor_pos = None
        self.banner_h = 40

        # Calculate scale factors for resizing
        self.scale_factor_x = 800 / first_frame.shape[1]
        self.scale_factor_y = 600 / first_frame.shape[0]

        self._initialize_roi_data()

    def _initialize_roi_data(self):
        """Initialize ROI data structure based on user input"""
        for i in range(self.n_rois):
            print(f"\n=== ROI {i+1} Configuration ===")
            n_stripes = int(
                input(
                    f"How many stripes for ROI {i+1}? (Enter an integer; stripes must be consecutive)\n> "
                )
            )
            n_lanes = int(
                input(
                    f"How many lanes for ROI {i+1}? (Enter an integer; lanes must be consecutive)\n> "
                )
            )

            self.roi_data.append(
                {
                    "id": i,
                    "n_stripes": n_stripes,
                    "n_lanes": n_lanes,
                    "points": [],
                    "endpoints": [],
                    "lane_pts": [],
                    "mode": "STRIPES",
                    "roi_pts": None,
                    "M": None,
                    "inv_M": None,
                    "dst_width": None,
                    "dst_length": None,
                    "color": ROI_COLORS[i % len(ROI_COLORS)],
                    "completed": False,
                }
            )

    def screen_to_original(self, px, py):
        """Convert screen coordinates to original frame coordinates"""
        return int(px / self.scale_factor_x), int(py / self.scale_factor_y)

    def original_to_screen(self, px, py):
        """Convert original frame coordinates to screen coordinates"""
        return int(px * self.scale_factor_x), int(py * self.scale_factor_y)

    def _banner_text(self):
        """Generate banner text based on current selection state"""
        if self.current_roi >= len(self.roi_data):
            return "All ROIs configured! SPACE starts video | ESC exits"

        roi = self.roi_data[self.current_roi]
        roi_prefix = f"ROI {self.current_roi + 1}/{self.n_rois}: "

        if roi["mode"] == "STRIPES":
            return roi_prefix + "Click the START and END points of the stripe sequence"
        if roi["mode"] == "ENDPOINTS":
            return (
                roi_prefix
                + f"Click Road {'Start' if len(roi['endpoints'])==0 else 'End'} ({len(roi['endpoints'])}/2)"
            )
        if roi["mode"] == "LANE":
            return (
                roi_prefix
                + f"Click rectangle corners TL,TR,BR,BL ({len(roi['lane_pts'])}/4)"
            )
        return "SPACE starts video | ESC exits"

    def _on_mouse(self, event, x, y, flags, param):
        """Handle mouse events for ROI selection"""
        # Convert screen coordinates to original frame coordinates
        orig_x, orig_y = self.screen_to_original(x, y)
        self.cursor_pos = (orig_x, orig_y)

        if (
            y < self.banner_h
            or event != cv.EVENT_LBUTTONDOWN
            or self.current_roi >= len(self.roi_data)
        ):
            return

        roi = self.roi_data[self.current_roi]

        if roi["mode"] == "STRIPES" and len(roi["points"]) < 2:
            roi["points"].append((orig_x, orig_y))
        elif roi["mode"] == "ENDPOINTS" and len(roi["endpoints"]) < 2:
            roi["endpoints"].append((orig_x, orig_y))
        elif roi["mode"] == "LANE" and len(roi["lane_pts"]) < 4:
            roi["lane_pts"].append((orig_x, orig_y))
            if len(roi["lane_pts"]) == 4:
                roi["completed"] = True
                self.current_roi += 1

    def _draw_visualization(self, vis):
        """Draw all ROI visualization elements"""
        # Draw all completed ROIs
        for roi_idx, roi in enumerate(self.roi_data):
            roi_color = roi["color"]

            # Draw points for current ROI being configured
            if roi_idx == self.current_roi and roi["mode"] == "STRIPES":
                for i, (px, py) in enumerate(roi["points"]):
                    sx, sy = self.original_to_screen(px, py)
                    cv.circle(vis, (sx, sy), 4, COL_RED, -1)
                    cv.putText(vis, f"{i+1}", (sx + 6, sy - 6), FONT, 0.5, COL_RED, 1)

            # Draw endpoints for all ROIs
            for i, (sx, sy) in enumerate(roi["endpoints"]):
                disp_x, disp_y = self.original_to_screen(sx, sy)
                cv.circle(vis, (disp_x, disp_y), 5, COL_GRN, -1)
                cv.putText(
                    vis,
                    f"R{roi_idx+1}-{'S' if i == 0 else 'E'}",
                    (disp_x + 6, disp_y - 6),
                    FONT,
                    0.5,
                    COL_GRN,
                    1,
                )

            # Draw lane points for all ROIs
            for i, (rx, ry) in enumerate(roi["lane_pts"]):
                disp_x, disp_y = self.original_to_screen(rx, ry)
                cv.circle(vis, (disp_x, disp_y), 5, roi_color, -1)
                cv.putText(
                    vis,
                    f"R{roi_idx+1}-{i+1}",
                    (disp_x + 6, disp_y - 6),
                    FONT,
                    0.4,
                    roi_color,
                    1,
                )

            # Draw helper lines for current ROI
            if (
                roi_idx == self.current_roi
                and roi["mode"] == "LANE"
                and len(roi["endpoints"]) == 2
            ):
                self._draw_helper_lines(vis, roi)

    def _draw_helper_lines(self, vis, roi):
        """Draw helper lines for lane point selection"""
        end_pt = np.array(roi["endpoints"][1])

        if len(roi["lane_pts"]) == 0 and self.cursor_pos is not None:
            cur = np.array(self.cursor_pos)
            dv = end_pt - cur
            if np.linalg.norm(dv) > 0:
                dv = dv / np.linalg.norm(dv)
                start_point = self.original_to_screen(*(cur - 2000 * dv).astype(int))
                end_point = self.original_to_screen(*(cur + 2000 * dv).astype(int))
                cv.line(vis, start_point, end_point, COL_HELPER, 1, cv.LINE_AA)

        if len(roi["lane_pts"]) == 1:
            p0 = np.array(roi["lane_pts"][0])
            dv = end_pt - p0
            dv = dv / np.linalg.norm(dv)
            start_point = self.original_to_screen(*(p0 - 2000 * dv).astype(int))
            end_point = self.original_to_screen(*(p0 + 2000 * dv).astype(int))
            cv.line(vis, start_point, end_point, COL_HELPER, 1, cv.LINE_AA)

        if len(roi["lane_pts"]) == 2 or len(roi["lane_pts"]) == 3:
            tl, tr = np.array(roi["lane_pts"][0]), np.array(roi["lane_pts"][1])
            start_pt = np.array(roi["endpoints"][0])
            ed = tr - tl
            ed = ed / np.linalg.norm(ed)
            start_point = self.original_to_screen(*(start_pt - 2000 * ed).astype(int))
            end_point = self.original_to_screen(*(start_pt + 2000 * ed).astype(int))
            cv.line(vis, start_point, end_point, COL_HELPER, 1, cv.LINE_AA)

    def _handle_space_key(self):
        """Handle space key press for advancing ROI configuration"""
        if self.current_roi < len(self.roi_data):
            roi = self.roi_data[self.current_roi]
            if roi["mode"] == "STRIPES" and len(roi["points"]) == 2:
                self._interpolate_stripes(roi)
            elif roi["mode"] == "ENDPOINTS" and len(roi["endpoints"]) == 2:
                roi["mode"] = "LANE"
        elif all(roi["completed"] for roi in self.roi_data):
            return True  # Signal that selection is complete
        return False

    def _interpolate_stripes(self, roi):
        """Interpolate stripe positions between start and end points"""
        start_pt, end_pt = np.array(roi["points"][0]), np.array(roi["points"][1])
        stripe_vec = end_pt - start_pt
        stripe_vec = stripe_vec / np.linalg.norm(stripe_vec)
        stripe_total_len = (
            roi["n_stripes"] * LANE_LEN_M + (roi["n_stripes"] - 1) * GAP_LEN_M
        )
        unit_px = np.linalg.norm(end_pt - start_pt) / stripe_total_len
        stripe_px_len = LANE_LEN_M * unit_px
        gap_px_len = GAP_LEN_M * unit_px

        generated_points = []
        cursor = np.array(start_pt, dtype=np.float32)
        for _ in range(roi["n_stripes"]):
            p1 = cursor.copy()
            p2 = p1 + stripe_vec * stripe_px_len
            generated_points.append(tuple(p1.astype(int)))
            generated_points.append(tuple(p2.astype(int)))
            cursor = p2 + stripe_vec * gap_px_len

        roi["points"] = generated_points
        roi["endpoints"] = [roi["points"][0], roi["points"][-1]]
        roi["mode"] = "LANE"

    def select_rois(self):
        """Main method to run ROI selection interface"""
        cv.namedWindow("Select")
        cv.setMouseCallback("Select", self._on_mouse)
        sel_img = self.first_frame.copy()

        # UI Loop
        while self.current_roi < len(self.roi_data) or not all(
            roi["completed"] for roi in self.roi_data
        ):
            # Create a resized version for display
            sel_img_resized = cv.resize(sel_img, (800, 600))
            vis = sel_img_resized.copy()

            # Add banner
            cv.rectangle(vis, (0, 0), (vis.shape[1], self.banner_h), COL_GREY, -1)
            cv.putText(
                vis,
                self._banner_text(),
                (10, 28),
                FONT,
                0.7,
                (255, 255, 255),
                2,
                cv.LINE_AA,
            )

            # Draw visualization elements
            self._draw_visualization(vis)

            cv.imshow("Select", vis)
            k = cv.waitKey(20) & 0xFF

            if k == 27:  # ESC key
                cv.destroyWindow("Select")
                return None  # User cancelled

            if k == ord(" "):  # Space key
                if self._handle_space_key():
                    break

        cv.destroyWindow("Select")
        return self.roi_data

    def calculate_perspective_transforms(self, pixels_per_meter=30):
        """Calculate perspective transform matrices for all completed ROIs"""
        for roi in self.roi_data:
            if roi["completed"]:
                # Sort the ROI points to ensure they are in order: TL, TR, BR, BL
                pts = np.array(roi["lane_pts"], np.float32)
                roi_pts = pts[
                    np.argsort(
                        np.arctan2(
                            pts[:, 1] - pts.mean(0)[1], pts[:, 0] - pts.mean(0)[0]
                        )
                    )
                ].astype(np.int32)
                roi["roi_pts"] = roi_pts

                # Define real-world dimensions
                real_roi_length = (
                    roi["n_stripes"] * LANE_LEN_M + (roi["n_stripes"] - 1) * GAP_LEN_M
                )
                real_roi_width = roi["n_lanes"] * LANE_WIDTH_M

                # Define the destination points for the bird's eye view
                dst_width = int(real_roi_width * pixels_per_meter)
                dst_length = int(real_roi_length * pixels_per_meter)
                roi["dst_width"] = dst_width
                roi["dst_length"] = dst_length

                # Define destination points (bird's eye view)
                dst_pts = np.array(
                    [
                        [0, 0],  # Top left
                        [dst_width, 0],  # Top right
                        [dst_width, dst_length],  # Bottom right
                        [0, dst_length],  # Bottom left
                    ],
                    dtype=np.float32,
                )

                # Get the perspective transform matrices
                roi["M"] = cv.getPerspectiveTransform(
                    roi_pts.astype(np.float32), dst_pts
                )
                roi["inv_M"] = cv.getPerspectiveTransform(
                    dst_pts, roi_pts.astype(np.float32)
                )
