import os

import torchvision.ops
from ultralytics import YOLO
import cv2
import numpy as np
import torch
import argparse


def preprocess_cv2(img_path, img_size=640):
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (img_size, img_size))
    img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
    return img_tensor.unsqueeze(0), img  # model input, original BGR image

# Draw bounding boxes
def draw_boxes_cv2(image, boxes, class_names):
    for box in boxes:
        c1, c2, w, h, conf, cls_id = box
        x1 = c1 - w // 2
        y1 = c2 - h // 2
        x2 = c1 + w // 2
        y2 = c2 + h // 2
        label = f"{class_names[int(cls_id)]}: {conf:.2f}"
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(image, label, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1)
    return image

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and extract COCO dataset.")
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="image to perform inference."
    )
    parser.add_argument(
        "--weights",
        type=str,
        required=False,
        default="",
        help="model_path"
    )
    args = parser.parse_args()

    class_names = ['human']
    if args.model_path!= '':
        model = YOLO(args.model_path)

    else:
        tmp_dir = './tmp'
        os.makedirs(tmp_dir, exist_ok=True)
        model = YOLO(os.path.join(tmp_dir, 'yolo11s.pt'))

    input_tensor, original_img = preprocess_cv2(args.source_img)

    with torch.no_grad():
        preds = model(input_tensor)[0].boxes.to('cpu')
        boxes = torch.cat([preds.xywh, preds.conf.view(-1, 1), preds.cls.view(-1, 1)], dim=-1)[preds.cls==0].numpy()

    # Filter predictions
    conf_thresh = 0.3
    boxes = boxes[boxes[:, -2] > conf_thresh]

    # Draw and show/save
    result_img = draw_boxes_cv2(original_img, boxes, class_names)
    cv2.imshow('Inference', result_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()