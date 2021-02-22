import torch
import torchvision
from PIL import Image
import os
import numpy as np
import cv2
from tqdm import tqdm
import pandas as pd
from utils import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def get_person_box(boxes, labels, conf, conf_th = 0.3):
    person = []
    for i in range(len(boxes)):
        if labels[i] != 1 or conf[i] < conf_th:
            continue
        box = [boxes[i][j].item() for j in range(4)]
        box.append(conf[i].item())
        person.append(box)
    return person

def size(box):
    return (box[3]-box[1])*(box[2]-box[0])

def get_overlap(box1, box2):
    if box1[0] > box2[2] or box1[2] < box2[0] or box1[1] > box2[3] or box1[3] < box2[1]:
        return 0
    
    x1 = max(box1[0],box2[0])
    y1 = max(box1[1],box2[1])
    x2 = min(box1[2],box2[2])
    y2 = min(box1[3],box2[3])

    return size([x1,y1,x2,y2])/min(size(box1),size(box2))

def merge_box(box1, box2):
    x1 = min(box1[0],box2[0])
    y1 = min(box1[1],box2[1])
    x2 = max(box1[2],box2[2])
    y2 = max(box1[3],box2[3])
    return [x1,y1,x2,y2]

def merge(boxes, mns_th = 0.4):
    l = len(boxes)
    merged = [False for i in range(l)]
    ret = []
    for i in range(len(boxes)):
        merge = False
        for j in range(i+1, len(boxes)):
                if get_overlap(boxes[i], boxes[j]) > mns_th:
                    boxes[j] = merge_box(boxes[i],boxes[j])
                    merged[i] = True

    for i in range(l):
        if merged[i]:
            continue
        ret.append(boxes[i])
    return ret




model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True).to(device)

model.eval()

imglist = os.listdir('../train_imgs')
imglist.sort()


df = pd.DataFrame(columns=['image','x1','y1','x2','y2'])

for i,imgpath in enumerate(tqdm(imglist)):
    img = cv2.imread("../train_imgs/"+imgpath)
    img_np = np.array(img)/255
    img_ten = torch.from_numpy(img_np).permute(2,0,1).unsqueeze(0).float().to(device)
    
    result = dict(model(img_ten)[0])
    boxes = get_person_box(result['boxes'].cpu(),result['labels'].cpu(),result['scores'].cpu())
    boxes = merge(boxes)
    if len(boxes) == 0:
        print(imgpath)
    for box in boxes:
        x1 = int(box[0])
        y1 = int(box[1])
        x2 = int(box[2])
        y2 = int(box[3])
        df = df.append({'image':imgpath,"x1":x1,"y1":y1,"x2":x2,"y2":y2},ignore_index=True)

df.to_csv("detect.csv",index=False)    
