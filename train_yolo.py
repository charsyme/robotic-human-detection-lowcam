from ultralytics import YOLO
from pathlib import Path
import torch
import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and extract COCO dataset.")
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to the logs and weights will be saved."
    )
    parser.add_argument(
        "--yaml_fln",
        type=str,
        required=True,
        help="YOLO YAML file"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=False,
        choices=["YOLO11n", "YOLO11s"],
        default="YOLO11n",
        help="YOLO model."
    )
    args = parser.parse_args()

    yolo_urls = {
        'YOLO11n' : 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt',
        'YOLO11s': 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.pt'
    }
    filename = os.path.join(args.output_dir, args.model)
    os.makedirs(args.output_dir, exist_ok=True)
    model = YOLO(yolo_urls[args.model])
    model.train(data=args.yaml_fln, epochs=20, imgsz=640, batch=-1, save_period=1, project=args.output_dir)
    model_path = Path(os.path.join(args.output_dir,'best.pt'))
    torch.save(model.state_dict(), model_path)
