import math
from tqdm import tqdm
from PIL import Image
import os

import math
from tqdm import tqdm
from PIL import Image
import os

if __name__ == "__main__":
    ROI_path = '/home/anhnt596/LargeROIPalm/ROI_Palm_Full'
    Original_path = '/home/anhnt596/LargeROIPalm/data'

    for session in os.listdir(Original_path):
        session_path = os.path.join(Original_path, session)
        for i in tqdm(range(1, 6001)):
            personID = math.ceil(i/10) + 162 
            image_order = i%10 + 11 if session == "session2" else i%10 + 1
            
            folder_path = os.path.join(ROI_path, f"{int(personID):06d}")
            if not os.path.exists(folder_path):
                os.mkdir(folder_path)
            save_path = os.path.join(folder_path, f"{int(image_order):03d}.jpg")
            
            image_name = f"{int(i):05d}.bmp"
            image_path = os.path.join(session_path, image_name)

            image = Image.open(image_path)
            image = image.resize((224,224))
            image.save(save_path, "JPEG")



    





    