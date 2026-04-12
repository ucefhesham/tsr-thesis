from ultralytics import YOLO
import os

def prepare_gtsdb_yaml():
    """Generates the GTSDB dataset configuration file for YOLO if it doesn't exist."""
    yaml_path = "data/gtsdb.yaml"
    
    if os.path.exists(yaml_path):
        print(f"Using existing dataset configuration at {yaml_path}")
        return yaml_path
        
    os.makedirs("data", exist_ok=True)
    
    # We define the 43 standard GTSRB classes
    classes = [
        "speed_limit_20", "speed_limit_30", "speed_limit_50", "speed_limit_60", 
        "speed_limit_70", "speed_limit_80", "end_of_speed_limit_80", "speed_limit_100", 
        "speed_limit_120", "no_overtaking", "no_overtaking_trucks", "right_of_way_at_next_intersection", 
        "priority_road", "yield", "stop", "no_vehicles", "no_vehicles_heavy", "no_entry", 
        "general_danger", "curve_left", "curve_right", "double_curve", "bumpy_road", 
        "slippery_road", "road_narrows_right", "road_works", "traffic_signals", 
        "pedestrians", "children", "bicycles", "ice_snow", "wild_animals", "end_of_all_limits", 
        "turn_right_ahead", "turn_left_ahead", "ahead_only", "ahead_or_right", "ahead_or_left", 
        "keep_right", "keep_left", "roundabout_mandatory", "end_of_no_overtaking", "end_of_no_overtaking_trucks"
    ]
    
    yaml_content = f"""
# GTSDB YOLO format dataset config
path: ../data/external/gtsdb  # dataset root dir
train: images/train  # train images (relative to 'path')
val: images/val  # val images (relative to 'path')

# Classes
names:
"""
    for i, cls_name in enumerate(classes):
        yaml_content += f"  {i}: {cls_name}\n"
        
    with open(yaml_path, "w") as f:
        f.write(yaml_content)
        
    print(f"Generated default GTSDB yaml config at {yaml_path}")
    print("WARNING: Please ensure your data/external/gtsdb folder contains 'images/train', 'images/val' and 'labels/' directories in YOLO format.")
    return yaml_path

def main():
    print("--- Initializing YOLOv11 GTSDB Fine-Tuning ---")
    
    yaml_path = prepare_gtsdb_yaml()
    
    # Load the base COCO model
    model = YOLO("yolo11n.pt")
    
    # Fine-tune on GTSDB
    # Note: 20 epochs is usually sufficient for transfer learning YOLO onto traffic signs 
    # to achieve reasonable Region Proposals for our downstream pipeline.
    print("Starting Training (Transfer Learning)...")
    try:
        results = model.train(
            data=yaml_path,
            epochs=20,
            imgsz=640,
            batch=16,
            device="cuda", # Automatically uses GPU if available
            project="logs/detect",
            name="gtsdb_ft",
            exist_ok=True
        )
        print("\n--- Training Complete ---")
        print("Best Weights saved to: logs/detect/gtsdb_ft/weights/best.pt")
        print("Please update detector.py to point to this path before running eval.py!")
    except Exception as e:
        print(f"\n[CRITICAL ERROR] Training failed: {e}")
        print("Have you downloaded the GTSDB dataset in YOLO format into data/external/gtsdb?")
        print("Tip: You can easily download it from Roboflow Universe (search: 'GTSDB YOLOv8').")

if __name__ == "__main__":
    main()
