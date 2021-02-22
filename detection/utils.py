import torch
def plot_boxes_cv2(img, boxes, points, savename=None, class_names=None, color=None):
    import cv2
    colors = torch.FloatTensor([[1, 0, 1], [0, 0, 1], [0, 1, 1], [0, 1, 0], [1, 1, 0], [1, 0, 0]])

    def get_color(c, x, max_val):
        ratio = float(x) / max_val * 5
        i = int(math.floor(ratio))
        j = int(math.ceil(ratio))
        ratio = ratio - i
        r = (1 - ratio) * colors[i][c] + ratio * colors[j][c]
        return int(r * 255)

    for i in range(1,len(points),2):
        img = cv2.circle(img, (int(points[i]),int(points[i+1])), radius=3, color=(0, 0, 255), thickness=-1)
    
    for box in boxes:
        x1 = int(box[0])
        y1 = int(box[1])
        x2 = int(box[2])
        y2 = int(box[3])


        if color:
            rgb = color
        else:
            rgb = (255, 0, 0)
        img = cv2.rectangle(img, (x1, y1), (x2, y2), rgb, 1)
    if savename:
        cv2.imwrite(savename, img)
    return img



def plot_box_cv2(img, box, points, savename=None, class_names=None, color=None):
    import cv2
    colors = torch.FloatTensor([[1, 0, 1], [0, 0, 1], [0, 1, 1], [0, 1, 0], [1, 1, 0], [1, 0, 0]])

    def get_color(c, x, max_val):
        ratio = float(x) / max_val * 5
        i = int(math.floor(ratio))
        j = int(math.ceil(ratio))
        ratio = ratio - i
        r = (1 - ratio) * colors[i][c] + ratio * colors[j][c]
        return int(r * 255)

    x1 = int(box[0])
    y1 = int(box[1])
    x2 = int(box[2])
    y2 = int(box[3])

    for i in range(1,len(points),2):
        img = cv2.circle(img, (int(points[i]),int(points[i+1])), radius=3, color=(0, 0, 255), thickness=-1)

    if color:
        rgb = color
    else:
        rgb = (255, 0, 0)
    img = cv2.rectangle(img, (x1, y1), (x2, y2), rgb, 1)
    if savename:
        cv2.imwrite(savename, img)
    return img

