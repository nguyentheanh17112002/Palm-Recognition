import os

import cv2
from tqdm import tqdm

if __name__ == "__main__":
    input_path = "/home/anhnt596/LargeROIPalm/GPDS/HandsGPDS150/manosGPDS"
    output_path = "/home/anhnt596/LargeROIPalm/ROI_Palm_Full"

    for id in tqdm(os.listdir(input_path)):
        id_path = os.path.join(input_path, id)
        roi_list = [
            image
            for image in os.listdir(id_path)
            if image.startswith("palma") and image.lower().endswith(("jpg", "jpeg"))
        ]
        for i in range(len(roi_list)):
            img_path = os.path.join(id_path, roi_list[i])
            img = cv2.imread(img_path)
            img = cv2.resize(img, (224, 224))

            save_folder = f"{int(id)+762:06d}"
            save_folder = os.path.join(output_path, save_folder)
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)

            name = f"{int(i)+1:03d}.jpg"

            save_path = os.path.join(save_folder, name)

            cv2.imwrite(save_path, img)
