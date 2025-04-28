import torch
import cv2
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_boxes
from utils.plots import Annotator, colors
import numpy as np

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = attempt_load('C:/Users/OS/yolov5/runs/train/exp10/weights/best.pt', device=device)
model.eval()

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    img = frame[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device).float()
    img /= 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    with torch.no_grad():
        pred = model(img)[0]

    pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)
    annotator = Annotator(frame, line_width=2, example=str(model.names))

    for det in pred:
        if len(det):
            det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], frame.shape).round()
            for *xyxy, conf, cls in reversed(det):
                c = int(cls)
                label = f'{model.names[c]} {conf:.2f}'
                annotator.box_label(xyxy, label, color=colors(c, True))
    cv2.imshow('YOLOv5', annotator.result())
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()