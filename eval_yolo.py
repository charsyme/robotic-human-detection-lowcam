from ultralytics import YOLO
import cv2
import numpy as np
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and extract COCO dataset.")
    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Path to the weights file."
    )
    parser.add_argument(
        "--yaml_fln",
        type=str,
        required=True,
        help="YOLO YAML file"
    )
    args = parser.parse_args()

    model = YOLO(args.weights)  # or "best.pt"
    metrics = model.val(data=args.yaml_fln, split="test")

    # Print evaluation results
    print(metrics)