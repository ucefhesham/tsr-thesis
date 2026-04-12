import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLO
from typing import List, Tuple, Optional
import numpy as np

class TrafficSignDetector(nn.Module):
    """
    Two-stage perception detector wrapping YOLOv11.
    Implements perfect square padding to preserve geographic aspect ratios 
    of detected signs before classification.
    """
    def __init__(self, model_path: str = "yolo11n.pt", conf_threshold: float = 0.25):
        super().__init__()
        self.detector = YOLO(model_path)
        self.conf_threshold = conf_threshold
        
        # GTSDB/Mapillary ID mapping to GTSRB 43 classes
        # This mapping is crucial for semantic consistency
        self.class_mapping = {i: i for i in range(43)} # Defaulting to identity for compatible sets
        
    def detect_crops(self, image_tensor: torch.Tensor) -> Tuple[List[torch.Tensor], List[dict]]:
        """
        Detects signs and extracts square-padded crops.
        
        Args:
            image_tensor: (3, H, W) float tensor [0, 1] or [0, 255]
            
        Returns:
            List of crops (padded squares) and their metadata (boxes, conf)
        """
        # Convert to numpy for YOLO if necessary, or pass tensor if supported
        # Ultralytics supports torch.Tensor [B, 3, H, W]
        img_input = image_tensor.unsqueeze(0) if image_tensor.dim() == 3 else image_tensor
        
        results = self.detector.predict(img_input, conf=self.conf_threshold, verbose=False)
        result = results[0]
        
        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        cls_ids = result.boxes.cls.cpu().numpy().astype(int)
        
        crops = []
        metadata = []
        
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            
            # Extract raw rectangular crop
            # Ensure dimensions are within bounds
            h_img, w_img = image_tensor.shape[1], image_tensor.shape[2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w_img, x2), min(h_img, y2)
            
            raw_crop = image_tensor[:, y1:y2, x1:x2]
            
            # Perfect Square Padding Logic
            _, h, w = raw_crop.shape
            if h == 0 or w == 0: continue
            
            side = max(h, w)
            diff = abs(h - w)
            p1 = diff // 2
            p2 = diff - p1
            
            if w > h: # Wider -> pad top/bottom
                padded_crop = F.pad(raw_crop, (0, 0, p1, p2), mode='constant', value=0)
            else: # Taller -> pad left/right
                padded_crop = F.pad(raw_crop, (p1, p2, 0, 0), mode='constant', value=0)
                
            crops.append(padded_crop)
            metadata.append({
                "box": [x1, y1, x2, y2],
                "conf": confs[i],
                "cls_id": cls_ids[i],
                "iou": 0.0 # To be filled by matching logic in eval loop
            })
            
        return crops, metadata

def load_european_weights(model: TrafficSignDetector, path: str):
    """Helper to load specific GTSDB or Mapillary fine-tuned weights."""
    model.detector = YOLO(path)
    print(f"Loaded European Perception Weights from: {path}")
