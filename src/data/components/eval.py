import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import cv2
import numpy as np
import math
import os

def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")

    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # return the ordered coordinates
    return rect


#function to transform image to four points
def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)

    # # multiply the rectangle by the original ratio
    # rect *= ratio

    (tl, tr, br, bl) = rect

    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    # return the warped image
    return warped

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

def preprocess_image(path, img_width, img_height):
    base_options = python.BaseOptions(model_asset_path='/home/anhnt596/Palm-Recognition/src/data/components/hand_landmarker.task')
    options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
    detector = vision.HandLandmarker.create_from_options(options)

    image = mp.Image.create_from_file(path)

    detection_result = detector.detect(image)

    hand_landmarks_list = detection_result.hand_landmarks[0]

    thumb_cmc = (int(hand_landmarks_list[1].x * img_width), int(hand_landmarks_list[1].y* img_height))
    top_left = (int(hand_landmarks_list[5].x * img_width), int(hand_landmarks_list[5].y* img_height))
    top_right = (int(hand_landmarks_list[17].x * img_width), int(hand_landmarks_list[17].y* img_height))


    bottom_left, bottom_right = find_bl_br(thumb_cmc, top_left, top_right, img_width, img_height)

    pts = [top_left, top_right, bottom_left, bottom_right]
    pts = np.array(pts)

    original_image = cv2.imread(path)
    crop = four_point_transform(original_image, pts)
    
    crop = cv2.resize(crop, (224, 224))
    return crop

def save_image(image, path, image_name):
    save_path = os.path.join(path, image_name)
    cv2.imwrite(save_path, image)


if __name__ == "__main__":
    Original_path = "/home/anhnt596/Palm-Recognition/data/COEP/Original"
    ROI_path = "/home/anhnt596/Palm-Recognition/data/COEP/ROI"
    
    Original_train_path = os.path.join(Original_path,"train")
    Original_test_path = os.path.join(Original_path, 'test')

    ROI_train_path = os.path.join(ROI_path, 'train')
    ROI_test_path = os.path.join(ROI_path, 'test')

    for image_name in os.listdir(Original_train_path):
        original_path = os.path.join(Original_train_path, image_name)

        processed_image = preprocess_image(original_path, 1600, 1200)
        save_image(processed_image, ROI_train_path, image_name)

    for image_name in os.listdir(Original_test_path):
        original_path = os.path.join(Original_test_path, image_name)

        processed_image = preprocess_image(original_path, 1600, 1200)
        save_image(processed_image, ROI_test_path, image_name)

