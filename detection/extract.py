import pandas as pd
import cv2
import tqdm

def get_mid_dist(box):
    x = (box[0]+box[2])/2
    y = (box[1]+box[3])/2
    return (x-960)**2+(y-540)**2

def get_max_len(box):
    x = (box[2]-box[0])
    y = (box[3]-box[1])
    return max(x,y)

def remove_background(boxes):
    ret = 0
    d = 0
    for i in range(len(boxes)):
        box = boxes[i]
        dn = get_max_len(box)
        if dn > d:
            ret = i
            d = dn
    
    return boxes[ret]

def check_valid(boxes, points):
    ret = True
    for img_name, point in points.items():
        imgname = point[0]
        if imgname not in boxes.keys():
            continue
        box = boxes[imgname]
        valid = True
        for i in range(len(point),2):
            if box[0] > point[i] or point[i] > box[2]:
                valid = False
    
        for i in range(1, len(point),2):
            if box[1] > point[i] or point[i] > box[3]:
                valid = False
        if not valid:
            print(imgname)
            plot_invalid(imgname, box, point)
            ret = False
    
    return ret


def expand_box(box, length=250):
    box[0] = max(0, box[0]-length)
    box[1] = max(0, box[1]-length)
    box[2] = min(1920, box[2]+length)
    box[3] = min(1080, box[3]+length)
    return box


def get_boxes(filepath):
    df = pd.read_csv(filepath).values
    img_list = list(set(df[i][0] for i in range(len(df))))
    boxes = {img_name:[] for img_name in img_list}

    for box in df:
        boxes[box[0]].append(box[1:])
    
    return img_list, boxes

def plot_invalid(imgname, box, points, savename="./tmp/", class_names=None, color=None):
    img = cv2.imread("../train_imgs/"+imgname)
    x1 = int(box[0])
    y1 = int(box[1])
    x2 = int(box[2])
    y2 = int(box[3])

    rgb = (255, 0, 0)
    for i in range(len(points),2):
        img = cv2.circle(img, (int(points[i]),int(points[i+1])), radius=3, color=(0, 0, 255), thickness=-1)
    img = cv2.rectangle(img, (x1, y1), (x2, y2), rgb, 1)
    if savename:
        cv2.imwrite(savename+imgname, img)
    return img

        
def plot_boxes(imgname, boxes, savename="./tmp/"):
    img = cv2.imread("../train_imgs/"+imgname)
    rgb = (255, 0, 0)
    for box in boxes:
        x1 = int(box[0])
        y1 = int(box[1])
        x2 = int(box[2])
        y2 = int(box[3])

        img = cv2.rectangle(img, (x1, y1), (x2, y2), rgb, 1)
    if savename:
        cv2.imwrite(savename+"boxes_"+imgname, img)
    return img

def inspect(imgname):
    box = boxes[imgname]
    for b in box:
        print(b)
    plot_boxes(imgname,[remove_background(box)])

def statistics(boxes):
    ratio = 0
    size = 0
    n = len(boxes)

    for imgname, box in boxes.items():
        x = box[2]-box[0]
        y = box[3]-box[1]
        size += x*y
        ratio += max(x/y,y/x)
    print("size : "+str(size/n))
    print("ratio : "+str(ratio/n))

def extract_train(boxes, points, out_size):
    label = pd.read_csv("../train_df.csv").columns
    crop_point = pd.DataFrame(columns=label)    

    for imgname, box in tqdm.tqdm(boxes.items()):
        img = cv2.imread("../train_imgs/"+imgname)
        point = points[imgname]
        x1 = int(box[0])
        y1 = int(box[1])
        x2 = int(box[2])
        y2 = int(box[3])
        x = x2-x1
        y = y2-y1
        img_crop = img[y1:y2,x1:x2]
        img_crop = cv2.resize(img_crop,(out_size,out_size))
        cv2.imwrite("../train_crop/"+imgname, img_crop)
        hl = {}
        kp = []
        for i in range(1,len(label),2):
            hl[label[i]] = (point[i-1]-x1)/x * out_size
        for i in range(2,len(label),2):
            hl[label[i]] = (point[i-1]-y1)/y * out_size
        hl['image'] = imgname
        crop_point=crop_point.append(hl , ignore_index=True)    
    crop_point.to_csv("../crop_df.csv",index=False)

def get_points():
    points = pd.read_csv("../train_df.csv").values
    points_dic = {point[0]:point[1:] for point in points}
    return  points_dic

if __name__ == "__main__":
    img_list, boxes = get_boxes("detect.csv")
    points = get_points()
    #inspect("593-1-3-28-Z134_B-0000017.jpg")
    for imgname, box in boxes.items():
        boxes[imgname] = expand_box(remove_background(box))
    #check_valid(boxes,points) valid check complete
    #statistics(boxes)
    extract_train(boxes, points,400)
