import numpy as np
from collections import defaultdict
from scipy.spatial.distance import cdist

# ByteTrack implementation
class BYTETracker:
    def __init__(self, track_thresh=0.5, track_buffer=30, match_thresh=0.8):
        self.tracked_tracks = []  # Tracks that are being tracked
        self.lost_tracks = []     # Tracks that are lost but still tracked for track_buffer frames
        self.track_id_count = 0
        self.track_thresh = track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
    
    def update(self, bboxes, scores, class_ids, frame=None):
        # Create new tracks from detections
        new_track_ids = []
        for i, (bbox, score, class_id) in enumerate(zip(bboxes, scores, class_ids)):
            if score < self.track_thresh:
                continue
                
            # Check if it matches an existing track
            matched = False
            if self.tracked_tracks:
                # Calculate IoU with existing tracks
                ious = np.array([self._calculate_iou(bbox, t['bbox']) for t in self.tracked_tracks])
                if ious.max() > self.match_thresh:
                    idx = ious.argmax()
                    matched = True
                    track_id = self.tracked_tracks[idx]['track_id']
                    # Update the track
                    self.tracked_tracks[idx]['bbox'] = bbox
                    self.tracked_tracks[idx]['score'] = score
                    self.tracked_tracks[idx]['class_id'] = class_id
                    self.tracked_tracks[idx]['time_since_update'] = 0
                    new_track_ids.append(track_id)
                    
            # Create new track if not matched
            if not matched:
                self.track_id_count += 1
                new_track = {
                    'track_id': self.track_id_count,
                    'bbox': bbox,
                    'score': score,
                    'class_id': class_id,
                    'time_since_update': 0
                }
                self.tracked_tracks.append(new_track)
                new_track_ids.append(self.track_id_count)
        
        # Update track status
        for t in self.tracked_tracks:
            if t['track_id'] not in new_track_ids:
                t['time_since_update'] += 1
        
        # Move tracks that haven't been updated to lost_tracks
        still_tracked = []
        for t in self.tracked_tracks:
            if t['time_since_update'] == 0:
                still_tracked.append(t)
            elif t['time_since_update'] <= self.track_buffer:
                self.lost_tracks.append(t)
        
        self.tracked_tracks = still_tracked
        
        # Remove old lost tracks
        self.lost_tracks = [t for t in self.lost_tracks if t['time_since_update'] <= self.track_buffer]
        
        # Return tracked objects
        tracks = []
        for t in self.tracked_tracks + self.lost_tracks:
            if t['time_since_update'] == 0 or self._is_valid_lost_track(t):
                x1, y1, w, h = t['bbox']
                tracks.append({
                    'track_id': t['track_id'],
                    'bbox': [x1, y1, x1+w, y1+h],  # convert to ltrb format
                    'active': t['time_since_update'] == 0,
                    'class_id': t['class_id']
                })
                
        return tracks
    
    def _calculate_iou(self, bbox1, bbox2):
        """Calculate IoU between two bounding boxes [x,y,w,h]"""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        # Convert to xmin, ymin, xmax, ymax
        xmin1, ymin1, xmax1, ymax1 = x1, y1, x1 + w1, y1 + h1
        xmin2, ymin2, xmax2, ymax2 = x2, y2, x2 + w2, y2 + h2
        
        # Calculate intersection area
        xx1 = max(xmin1, xmin2)
        yy1 = max(ymin1, ymin2)
        xx2 = min(xmax1, xmax2)
        yy2 = min(ymax1, ymax2)
        
        w = max(0, xx2 - xx1)
        h = max(0, yy2 - yy1)
        intersection = w * h
        
        # Calculate union area
        union = (w1 * h1) + (w2 * h2) - intersection
        
        return intersection / union if union > 0 else 0
    
    def _is_valid_lost_track(self, track):
        """Check if a lost track is still valid for considering"""
        return track['time_since_update'] <= self.track_buffer

# Create a Track class for compatibility with the original code
class Track:
    def __init__(self, track_data):
        self.track_id = track_data['track_id']
        self.bbox = track_data['bbox']  # ltrb format
        self.is_active = track_data['active']
        self.class_id = track_data['class_id']
    
    def is_confirmed(self):
        return self.is_active
    
    def to_ltrb(self):
        return self.bbox