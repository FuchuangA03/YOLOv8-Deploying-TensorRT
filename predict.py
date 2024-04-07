from models import TRTModule  # isort:skip
import argparse
from pathlib import Path

import cv2
import torch
import time

from config import CLASSES, COLORS
from models.torch_utils import det_postprocess
from models.utils import blob, letterbox, path_to_list


def predict(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    Engine = TRTModule(args.engine, device)
    H, W = Engine.inp_info[0].shape[-2:]
 
    # set desired output names order
    Engine.set_desired(['num_dets', 'bboxes', 'scores', 'labels'])

    bgr = args.imgs
    draw = bgr.copy()
    bgr, ratio, mean_ratio, dwdh = letterbox(bgr, (W, H))
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    tensor = blob(rgb, return_seg=False)
    dwdh = torch.asarray(dwdh * 2, dtype=torch.float32, device=device)
    tensor = torch.asarray(tensor, device=device)
    # inference
    start = time.time()
    data = Engine(tensor)
    end = time.time()
    print("推理用时{}s".format(end - start))

    bboxes, scores, labels = det_postprocess(data)
    bboxes -= dwdh
    bboxes /= ratio

    # 将labels中的数字根据映射转换成标签名
    class_id = labels.tolist()
    label_map = {'broke':0, 'circle':1, 'good':2, 'lose':3, 'uncovered':4}
    labels = [key for value in class_id for key, val in label_map.items() if val == value]

    for (bbox, score, label) in zip(bboxes, scores, labels):
        bbox = bbox.round().int().tolist()
        cls_id = int(label_map[label])
        cls = CLASSES[cls_id]
        color = COLORS[cls]
        x1, y1, w, h = bbox
        # Draw the bounding box on the image
        cv2.rectangle(draw, (int(x1), int(y1)), (int(w), int(h)), color, int(4 / mean_ratio))

        # Create the label text with class name and score
        label = f"{label}: {score:.2f}"

        # # Calculate the dimensions of the label text
        (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5 / mean_ratio, int(1 / mean_ratio))

        # Calculate the position of the label text
        label_x = x1
        label_y = y1 - 10 if y1 - 10 > label_height else y1 + 10

        # Draw a filled rectangle as the background for the label text
        cv2.rectangle(
            draw, (label_x, label_y - label_height), (label_x + label_width, label_y + label_height), color, cv2.FILLED
        )

        # Draw the label text on the image
        cv2.putText(draw, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5 / mean_ratio, (0, 0, 0), int(1 / mean_ratio), cv2.LINE_AA)

    # import pdb
    # pdb.set_trace()
    return draw, str(class_id)