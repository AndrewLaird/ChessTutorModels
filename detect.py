import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from utils.datasets import LoadImages
from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression, \
    scale_coords, xyxy2xywh
# delete utils.plots
from utils.torch_utils import select_device, time_synchronized


# we want to set defaults for everything
# and then we want to cut out saving the images
# but not saving the points
def detect(source='data/input_images', weights='cropModel.pt', output='data/cropped_images'):
    save_img = False
    view_img = 'false'
    save_txt = 'true'
    # defaulted from trainging
    imgsz = 640
    project = output
    device = ''
    conf_thres = 0.25
    iou_thres = 0.45
    save_conf = True
    agnostic_nms = False
    classes = list(range(12))

    result_data = []

    # Directories
    #save_dir = Path(project, exist_ok=True)  # increment run
    #(save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    device = select_device(device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    if half:
        model.half()  # to FP16


    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
    # Set Dataloader
    vid_path, vid_writer = None, None
    save_img = True
    dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=False)[0]#opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms)
        t2 = time_synchronized()

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            #save_path = str(save_dir / p.name)  # img.jpg
            #txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # clear the file if it exists
                #with open(txt_path + '.txt', 'w') as f:
                    #f.write('')
                # Write results
                result_data.append([])
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (int(cls), *xywh, float(conf)) if save_conf else (cls, *xywh)  # label format
                        result_data[i].append(line)
                        #) no need to write the results to file, we will just return them from this function
                        #with open(txt_path + '.txt', 'a') as f:
                           # f.write(('%g ' * len(line)).rstrip() % line + '\n')

    #if save_txt or save_img:
    #    s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        #print(f"Results saved to {save_dir}{s}")

    #print(f'Done. ({time.time() - t0:.3f}s)')
    return result_data

