from ultralytics import YOLO
import torch
import cv2
import numpy as np
from PIL import Image
from typing import List, Union, Tuple

class YOLODetector:
    def __init__(self, model_path: str = "yolo11n.pt", task: str = "detect"):
        """
        Wrapper for Ultralytics YOLOv11 to handle sign region proposals.
        
        Args:
            model_path: Path to the .pt weights (e.g., 'yolo11n.pt' or a custom trained sign detector).
            task: YOLO task (default 'detect').
        """
        self.model = YOLO(model_path, task=task)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def detect_and_crop(
        self, 
        image: Union[str, np.ndarray, Image.Image], 
        conf_threshold: float = 0.25
    ) -> List[dict]:
        """
        Detects objects in an image and returns list of crops and their metadata.
        
        Args:
            image: Input image (path, numpy array, or PIL).
            conf_threshold: Confidence threshold for detection.
            
        Returns:
            List[dict]: Each dict contains 'crop' (np.ndarray), 'bbox' (xyxy), and 'conf'.
        """
        results = self.model(image, conf=conf_threshold, verbose=False)
        output = []

        for r in results:
            orig_img = r.orig_img
            for box in r.boxes:
                # Extract coordinates
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0].cpu().item())
                
                # Extract crop
                x1, y1, x2, y2 = xyxy
                # Ensure coordinates are within image bounds
                h, w = orig_img.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                crop = orig_img[y1:y2, x1:x2]
                
                output.append({
                    "crop": crop,
                    "bbox": [x1, y1, x2, y2],
                    "conf": conf
                })
                
        return output

    def get_most_confident_sign(self, image: Union[str, np.ndarray]) -> Union[np.ndarray, None]:
        """
        Helper for single-sign benchmarks where we only care about the best proposal.
        """
        detections = self.detect_and_crop(image)
        if not detections:
            return None
            
        # Sort by confidence descending
        detections.sort(key=lambda x: x["conf"], reverse=True)
        return detections[0]["crop"]

# Usage example (simulated for thesis script)
if __name__ == "__main__":
    # This would typically be run with a model trained on a detection dataset like GTSDB
    detector = YOLODetector("yolo11n.pt")
    print("YOLO Detector wrapper initialized successfully.")
