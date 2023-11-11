from ultralytics import YOLO
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from dotenv import load_dotenv
import os

load_dotenv()

CV_PATH = os.environ.get('CV_PATH')

model = YOLO(CV_PATH)

def get_obj_size(box: np.ndarray, image: np.ndarray):
    x, y, w, h = box

    box_area = w * h
    image_area = image.shape[0] * image.shape[1]

    ratio = box_area / image_area

    if ratio < 0.001:
        return 'S'
    elif ratio < 0.07:
        return 'M'
    else:
        return 'L'


def create_overlay(coords: np.ndarray, image: np.ndarray):
    overlay = np.zeros((640, 640, 4), dtype=np.uint8)

    if coords.size > 0:
        for coord in coords:
            x_center, y_center, box_width, box_height = coord

            x1, y1 = int(x_center - box_width / 2), int(y_center - box_height / 2)
            x2, y2 = int(x_center + box_width / 2), int(y_center + box_height / 2)

            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255, 255), 2, cv2.LINE_AA)

            obj_size = get_obj_size(coord, image)

            pil_img = Image.fromarray(overlay)
            draw = ImageDraw.Draw(pil_img)

            font_size = {'S': int(image.shape[0] / 40), 'M': int(image.shape[0] / 25), 'L': int(image.shape[0] / 15)}[obj_size]
            font = ImageFont.truetype('Arial.ttf', font_size)

            text_x = x1 + 1
            text_y = y1

            draw.text((text_x, text_y), obj_size, font=font, fill=(255, 255, 255, 255))

            overlay = np.array(pil_img)

        overlay = cv2.resize(overlay, (image.shape[1], image.shape[0]))

    return overlay


def predict(image: np.ndarray):
    img = cv2.resize(image, (640, 640))

    results = model(img)
    for result in results:
        boxes = result.boxes
        conf = boxes.conf.numpy()
        all_coords = boxes.xywh.numpy()

    detections = np.where(conf > 0.7)
    coords = all_coords[detections]

    overlay = create_overlay(coords, image)

    return overlay
