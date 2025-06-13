import os
import cv2
from tqdm import tqdm
from PIL import Image


if __name__ == "__main__":
    input_path = '/home/anhnt596/LargeROIPalm/GPDS/HandsGPDS100Contactless2bands/GPDSmanosASM'
    output_path = '/home/anhnt596/LargeROIPalm/ROI_Palm_Full'

    for id in tqdm(os.listdir(input_path)):
        id_path = os.path.join(input_path, id)
        roi_list = [image for image in os.listdir(id_path) if image.startswith("Palma") and image.lower().endswith(('bmp'))]
        for i in range(len(roi_list)):
            img_path = os.path.join(id_path, roi_list[i])

            img = Image.open(img_path)
            img = img.resize((224,224))

            save_folder = f"{int(id)+912:06d}"
            save_folder = os.path.join(output_path, save_folder)
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)



            name = f'{int(i)+1:03d}.jpg'

            save_path = os.path.join(save_folder, name)

            img.save(save_path, "JPEG")
            
