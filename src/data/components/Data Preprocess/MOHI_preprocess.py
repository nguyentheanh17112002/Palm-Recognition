import numpy as np
import cv2
import math
from tqdm import tqdm

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import re
import os


def Crop_ROI(img, points):
    rect = np.array(points, dtype=np.float32).reshape(-1,1,2)
    width = 224
    height = 224
    dst = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]], dtype="float32").reshape(-1,1,2)

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    crop = cv2.warpPerspective(img, M, (width, height))
    return crop

def find_bl_br(thumb_cmc, tl, tr, img_width, img_height):
    perpendicular_vector = (tr[1] - tl[1], tl[0] - tr[0])

    thumb_to_tl = math.sqrt((tl[0] - thumb_cmc[0])**2 + (tl[1] - thumb_cmc[1])**2)
    thumb_to_tr = math.sqrt((tr[0] - thumb_cmc[0])**2 + (tr[1] - thumb_cmc[1])**2)

    if (0 <= tl[0] - perpendicular_vector[0] <= img_width) and (0 <= tl[1] - perpendicular_vector[1] <= img_height):
        x = tl[0] - perpendicular_vector[0]
        y = tl[1] - perpendicular_vector[1]
        dist = math.sqrt((x- thumb_cmc[0])**2 + (y - thumb_cmc[1])**2)

        if(dist > thumb_to_tl):
            bl = (tl[0] + perpendicular_vector[0],tl[1] + perpendicular_vector[1])
        else:
            bl = (x , y)
    else:
        bl = (tl[0] + perpendicular_vector[0],tl[1] + perpendicular_vector[1])

    if (0 <= tr[0] - perpendicular_vector[0] <= img_width) and (0 <= tr[1] - perpendicular_vector[1] <= img_height):
        x = tr[0] - perpendicular_vector[0]
        y = tr[1] - perpendicular_vector[1]
        dist = math.sqrt((x- thumb_cmc[0])**2 + (y - thumb_cmc[1])**2)

        if(dist > thumb_to_tr):
            br = (tr[0] + perpendicular_vector[0],tr[1] + perpendicular_vector[1])
        else:
            br = (x , y)
    else:
        br = (tr[0] + perpendicular_vector[0],tr[1] + perpendicular_vector[1])

    return bl, br 

def preprocess_image(path):
    base_options = python.BaseOptions(model_asset_path='/home/anhnt596/LargeROIPalm/hand_landmarker.task')
    options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
    detector = vision.HandLandmarker.create_from_options(options)
    try:
        original_image = cv2.imread(path)
        img_height, img_width, _ = original_image.shape

        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    except:
        return None

    try:
        detection_result = detector.detect(image)
        hand_landmarks_list = detection_result.hand_landmarks[0]
    except (IndexError, AttributeError) as e:
        print(f"Error processing image {path}: {e}")
        return None

    thumb_cmc = (int(hand_landmarks_list[1].x * img_width), int(hand_landmarks_list[1].y* img_height))
    top_left = (int(hand_landmarks_list[5].x * img_width), int(hand_landmarks_list[5].y* img_height))
    top_right = (int(hand_landmarks_list[17].x * img_width), int(hand_landmarks_list[17].y* img_height))

    bottom_left, bottom_right = find_bl_br(thumb_cmc, top_left, top_right, img_width, img_height)

    pts = np.array([top_left, top_right, bottom_right, bottom_left])

    ROI = Crop_ROI(original_image, pts)
    return ROI

def preprocess_name(image_name:str):
    bonus = 0
    if image_name.startswith('S2'):
        bonus = 200
    if image_name.startswith('S3'):
        bonus = 400

    segments = image_name.split("-")

    minus = 0
    if int(segments[1][1:]) > 100:
        minus = 50

    id = int(segments[1][1:]) + 1612 + bonus - minus


    image_order = segments[-1].split(".")[0]  
    if image_order == '':
        # S1-P71-M-14-4-.jpg
        image_order = segments[-2]
    image_order = re.sub(r'\D','', image_order)
    try:
        image_order = int(image_order)
        return id, image_order
    except ValueError as e:
        print(image_name)
        print(f"Error: {e}. Image Order: {image_order}")

        raise SystemExit()
    
def save_image(ROIpath, ROI, image_name):
    personID, image_order = preprocess_name(image_name)
    save_name = f"{int(image_order):03d}.jpg"
    folderPath = os.path.join(ROIpath, f"{int(personID):06d}")

    if not os.path.exists(folderPath):
        os.makedirs(folderPath)

    save_path = os.path.join(folderPath, save_name)

    cv2.imwrite(save_path, ROI)

if __name__ == "__main__":
    Original_path = "/home/anhnt596/LargeROIPalm/mohi+wehi/MOHI"
    ROI_path = "/home/anhnt596/LargeROIPalm/ROI_Palm_Full"

    for image_name in tqdm(os.listdir(Original_path)):
        image_path = os.path.join(Original_path, image_name)
        ROI = preprocess_image(image_path)
        if ROI is not None:
            save_image(ROI_path, ROI, image_name)