import cv2
import numpy as np


class ROISelector:
    def __init__(self):
        self.current_polygon = []
        self.polygons = []
        self.polygon_np_arrays = []

    def _draw_polygon_callback(self, event, x, y, flags, param):
        """Callback function for mouse events during ROI selection."""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.current_polygon.append((x, y))
            print(f"Added point: ({x}, {y})")

    def select_rois(self, frame):
        """Allows the user to select Regions of Interest (ROIs) on the video frame."""
        display_frame = frame.copy()
        cv2.namedWindow("Select ROIs")
        cv2.setMouseCallback("Select ROIs", self._draw_polygon_callback)

        print("\n--- ROI Selection ---")
        print("Click to define polygon points.")
        print("Press ENTER to confirm current polygon.")
        print("Press ESC to finish selection.")

        while True:
            temp_display_frame = display_frame.copy()

            # Draw existing polygons
            for poly_np in self.polygon_np_arrays:
                cv2.polylines(temp_display_frame, [poly_np], isClosed=True, color=(0, 255, 0), thickness=2)

            # Draw current polygon being created
            if len(self.current_polygon) >= 2:
                cv2.polylines(temp_display_frame, [np.array(self.current_polygon, np.int32)], isClosed=False,
                              color=(0, 255, 255), thickness=1)
            if len(self.current_polygon) >= 3:
                cv2.polylines(temp_display_frame, [np.array(self.current_polygon, np.int32)], isClosed=True,
                              color=(0, 255, 0), thickness=1)

            # Draw points
            for pt in self.current_polygon:
                cv2.circle(temp_display_frame, pt, 3, (0, 0, 255), -1)

            cv2.imshow("Select ROIs", temp_display_frame)
            key = cv2.waitKey(1) & 0xFF

            if key == 13:  # ENTER key
                if len(self.current_polygon) >= 3:
                    self.polygons.append(self.current_polygon.copy())
                    self.polygon_np_arrays.append(np.array(self.current_polygon, np.int32))
                    print(f"Polygon {len(self.polygons)} saved.")
                    self.current_polygon.clear()
                else:
                    print("Minimum 3 points required to save polygon.")
            elif key == 27:  # ESC key
                break

        cv2.destroyWindow("Select ROIs")
        return self.polygons, self.polygon_np_arrays