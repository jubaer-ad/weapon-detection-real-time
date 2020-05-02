#imports
import cv2
import numpy as np
import argparse
import time

parser = argparse.ArgumentParser("Weapon detector using yolov3 on top of the opencv")
group = parser.add_mutually_exclusive_group()

group.add_argument("-v", "--video", help = "Detect from video", action = "store_true")
group.add_argument("-i", "--image", help = "Detect from image", action = "store_true")

parser.add_argument("--image_path", help = "Path to image", default = "test1.jpg")
parser.add_argument("--video_path", help = "Path to video", default = "test1.mp4")
args = parser.parse_args()


#loading yolov3
def load_yolo():
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    classes = []
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return net, classes, output_layers

#load image
def load_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, None, fx = 1, fy = 1)
    height, weight, depth = img.shape
    return img, height, weight, depth

#preprocess
def preprocess(img, net, output_layers):
    blob = cv2.dnn.blobFromImage(img, scalefactor = 0.00392, size = (512, 512), mean = (0, 0, 0), swapRB = True, crop = False)
    net.setInput(blob)
    outputs = net.forward(output_layers)
    return blob, outputs

#bounding boxes info
def get_box_dimensions(outputs, height, width):
    boxes = []
    confs = []
    class_ids = []
    for output in outputs:
        for detect in output:
            scores = detect[5:]
            class_id = np.argmax(scores)
            conf = scores[class_id]
            if conf > 0.3:
                center_x = int(detect[0] * width)
                center_y = int(detect[1] * height)
                w = int(detect[2] * width)
                h = int(detect[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x , y, w, h])
                confs.append(float(conf))
                class_ids.append(class_id)
    return boxes, confs, class_ids


#drawing boxes
def draw_boxes(boxes, confs, class_ids, classes, img):
    indexes = cv2.dnn.NMSBoxes(boxes, confs, 0.5, 0.4)
    font = cv2.FONT_HERSHEY_PLAIN
    for i in indexes:
        x, y, w, h = boxes[int(i)]
        label = str(classes[class_ids[int(i)]])
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(img, label, (x, y - 5), font, 1, (100, 100, 255), 2)
    cv2.imshow("stfo", img)

#image detection
def detection_image(img_path):
    model, classes, output_layers = load_yolo()
    image, height, width, depth = load_image(img_path)
    blob, outputs = preprocess(image, model, output_layers)
    boxes, confs, class_ids = get_box_dimensions(outputs, height, width)
    draw_boxes(boxes, confs, class_ids, classes, image)
    while True:
        key = cv2.waitKey(1)
        if key == 27:
            break

#video detection
def detection_video(video_path):
    model, classes, output_layers = load_yolo()
    cap = cv2.VideoCapture(video_path)
    while True:
        _, frame = cap.read()
        height, width, channels = frame.shape
        blob, outputs = preprocess(frame, model, output_layers)
        boxes, confs, class_ids = get_box_dimensions(outputs, height, width)
        draw_boxes(boxes, confs, class_ids, classes, frame)
        key = cv2.waitKey(1)
        if key == 27:
            break
    cap.release()

if __name__ == '__main__':
        if args.video:
                video_path = args.video_path
                print("Opening " + video_path + ".....")
                detection_video(video_path)
        if args.image:
                img_path = args.image_path
                print("Opening " + img_path + " .....")
                detection_image(img_path)
        cv2.destroyAllWindows()







"""
#showing blobs
for b in blob:
    for n, img_blob in enumerate(b):
        cv2.imshow(str(n), img_blob)
"""
