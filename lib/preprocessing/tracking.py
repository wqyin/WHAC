import cv2
import numpy as np
import scipy.signal as signal
from ultralytics import YOLO
from sort.tracker import SortTracker

MINIMUM_FRMAES = 30

class YOLOPersonTracker:
    def __init__(self, cap, output_path="output.avi", model_path="./pretrained_models/yolov8x.pt"):
        """
        Initializes the YOLO-based person tracker.

        :param cap: Opened cv2.VideoCapture object.
        :param output_path: Path to save the output video.
        :param model_path: Path to the YOLO model weights.
        """
        self.model = YOLO(model_path)  # Load YOLO model
        self.tracker = SortTracker()  # Initialize SORT tracker
        self.cap = cap  # Use the provided video capture object

        # Get video properties for saving
        self.frame_width = int(cap.get(3))
        self.frame_height = int(cap.get(4))
        self.fps = int(cap.get(cv2.CAP_PROP_FPS))

        # Define video writer
        self.out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"XVID"), 
                                   self.fps, (self.frame_width, self.frame_height))
                                   
    def xyxy_to_xywh(self, bbox):
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1

        return x1, y1, w, h
    
    def xyxy_to_cxcys(self, bbox, s_factor=1.05):
        x1, y1, x2, y2 = bbox        
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        scale = max(x2 - x1, y2 - y1) / 200 * s_factor

        return np.array([[cx, cy, scale]])

    def process_frame(self, frame):
        """
        Runs YOLO detection on the frame and tracks detected people.

        :param frame: Input image frame.
        :return: Frame with tracking annotations.
        """
        results = self.model(frame, verbose=False)  # Run YOLO detection

        detections = []
        for r in results:
            for box in r.boxes:
                cls = int(box.cls.item())  # Class ID
                if cls == 0:  # "person" class in COCO dataset
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    score = box.conf.item()  # Confidence score
                    detections.append([x1, y1, x2, y2, score, 0])
        
        # Convert detections to numpy array
        detections = np.array(detections) if detections else np.empty((0, 5))

        # Update tracker with new detections
        tracked_objects = self.tracker.update(detections, 0)

        frame_results = {}
        # Draw tracking results
        for obj in tracked_objects:
            x1, y1, x2, y2, track_id, track_cls, track_conf = map(int, obj)
            
            if track_cls == 0: # only draw "person" class
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"ID {track_id}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                frame_results[track_id] = [x1, y1, x2, y2, track_conf]
        
        return frame, frame_results

    def run(self, save_output=False):
        """
        Runs the video processing loop and saves the output.
        """
        frames = []
        video_results = {}
        filtered_video_result = {}
        frame_id = 0
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            frame, tracked_results = self.process_frame(frame)
            video_results[frame_id] = tracked_results

            if save_output:
                self.out.write(frame)  # Save frame to output video

            frame_id += 1
            frames.append(frame)

        for frame_id, result in video_results.items():
            for tracked_id, bbox in result.items():
                if tracked_id not in filtered_video_result:
                    filtered_video_result[tracked_id] = {'frame_id': [], 'bbox_xyxy': [], 
                                                    'bbox_cxcys': [], 'bbox_xywh': []}

                filtered_video_result[tracked_id]['frame_id'].append(frame_id)
                filtered_video_result[tracked_id]['bbox_xyxy'].append(bbox[:4])
                filtered_video_result[tracked_id]['bbox_cxcys'].append(self.xyxy_to_cxcys(bbox[:4]))
                filtered_video_result[tracked_id]['bbox_xywh'].append(self.xyxy_to_xywh(bbox[:4]))
        
        ids = list(filtered_video_result.keys())
        for _id in ids:
            if len(filtered_video_result[_id]['frame_id']) < MINIMUM_FRMAES:
                del filtered_video_result[_id]
                continue
            kernel = int(int(self.fps/2) / 2) * 2 + 1
            smoothed_bbox = np.array([signal.medfilt(param, kernel) for param in np.array(filtered_video_result[_id]['bbox_cxcys']).T]).T
            smoothed_bbox_xyxy = np.array([signal.medfilt(param, kernel) for param in np.array(filtered_video_result[_id]['bbox_xyxy']).T]).T
            filtered_video_result[_id]['bbox_cxcys'] = smoothed_bbox.tolist()
            filtered_video_result[_id]['bbox_xyxy'] = smoothed_bbox_xyxy.tolist()

        self.cap.release() # Close video writer
        self.out.release()
        cv2.destroyAllWindows()

        return frames, filtered_video_result

# Usage example
if __name__ == "__main__":
    cap = cv2.VideoCapture("./demo/skateboard/city07.mp4")  # Replace with 0 for webcam
    tracker = YOLOPersonTracker(cap, output_path="./tracking_results.mp4")
    tracker.run(save_output=True)
