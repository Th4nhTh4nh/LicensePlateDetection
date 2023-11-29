"""
# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import util

# định nghĩa các đường dẫn đến cfg, weights , nameclass của model
model_cfg_path = os.path.join('.', 'model', 'cfg', 'yolov3-tiny.cfg')
model_weights_path = os.path.join('.', 'model', 'weights', 'yolov3-tiny_15000.weights')
class_names_path = os.path.join('.', 'model', 'class.names')

input_dir = "./pic" # đường dẫn đến thư mục ảnh
for img_name in os.listdir(input_dir):
    img_path = os.path.join(input_dir,img_name)


    # load các tên class của mình
    with open(class_names_path, 'r') as f:
        class_names = [j[:-1] for j in f.readlines() if len(j) > 2]
        f.close()

    # load model
    net = cv2.dnn.readNetFromDarknet(model_cfg_path, model_weights_path)

    # load image

    img = cv2.imread(img_path)

    H, W, _ = img.shape

    # chuyển đổi kích thước hình ảnh sao cho phù hợp với mạng YOLO đã cài đặt
    blob = cv2.dnn.blobFromImage(img, 1 / 255, (416, 416), (0, 0, 0), True)

    # get detections
    net.setInput(blob)

    detections = util.get_outputs(net)

    # bboxes, class_ids, confidences
    bboxes = []
    class_ids = []
    scores = []


    for detection in detections:
        # [x1, x2, x3, x4, x5, x6]
        bbox = detection[:4]

        xc, yc, w, h = bbox
        bbox = [int(xc * W), int(yc * H), int(w * W), int(h * H)]

        bbox_confidence = detection[4]

        class_id = np.argmax(detection[5:])
        score = np.amax(detection[5:])

        bboxes.append(bbox)
        class_ids.append(class_id)
        scores.append(score)

    #  loại bỏ các bbox có độ chính xác ko cao và giữ lại 1 bbox
    bboxes, class_ids, scores = util.NMS(bboxes, class_ids, scores)

    for bbox_, bbox in enumerate(bboxes):
        xc, yc, w, h = bbox

        # lấy tọa độ của biển số xe
        license_plate = img[int(yc - (h / 2)):int(yc + (h / 2)), int(xc - (w / 2)):int(xc + (w / 2)), :].copy()
        img = cv2.rectangle(img,
                            (int(xc - (w / 2)), int(yc - (h / 2))),
                            (int(xc + (w / 2)), int(yc + (h / 2))),
                            (0, 255, 0),
                            10)
        # xử lý ảnh biển số xe trước khi nhận diện kí tự
        license_plate_gray = cv2.cvtColor(license_plate,cv2.COLOR_BGR2GRAY)
        _,license_plate_thresh = cv2.threshold(license_plate_gray,103,255,cv2.THRESH_BINARY_INV)

    plt.figure()
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    # plt.figure()
    # plt.imshow(cv2.cvtColor(license_plate, cv2.COLOR_BGR2RGB))
    #
    # plt.figure()
    # plt.imshow(cv2.cvtColor(license_plate_gray, cv2.COLOR_BGR2RGB))


    plt.figure()
    plt.imshow(cv2.cvtColor(license_plate_thresh, cv2.COLOR_BGR2RGB))

    plt.show()
