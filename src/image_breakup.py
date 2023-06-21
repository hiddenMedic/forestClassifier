import cv2
import os
from tqdm import tqdm

"""
BASE_PATH = "data\\image_data\\useful\\"
FILE_NAME = "T19TFL_20230514T153559_TCI_10m.jp2"
FOLDER_NAME = "S2B_MSIL2A_20230514T153559_N0509_R111_T19TFL_20230514T195610"
SAVE_PATH = "proc_data\\pieces\\square\\"
"""
BASE_PATH = "data\\image_data\\raw\\_serbia\\S2A_MSIL2A_20230613T094041_N0509_R036_T34TCR_20230613T135353\\GRANULE\\a\\IMG_DATA\\R10m\\"
FILE_NAME = "T34TCR_20230613T094041_TCI_10m.jp2"
FOLDER_NAME = "S2A_MSIL2A_20230613T094041_N0509_R036_T34TCR_20230613T135353"
SAVE_PATH = "proc_data\\pieces\\square\\serbia\\"

os.mkdir(SAVE_PATH + FOLDER_NAME)

img = cv2.imread(BASE_PATH + FILE_NAME)
height = img.shape[0]
width = img.shape[1]

for i in tqdm(range(width // 100)):
    for j in range(height // 100):
        sim = img[(j*100):((j+1)*100), (i*100):((i+1)*100)]
        cv2.imwrite(SAVE_PATH + FOLDER_NAME + f"\\img{i*100}_{j*100}.png", sim)
        
f = open(SAVE_PATH + FOLDER_NAME + "\\f_name.txt", "w")
f.write(FILE_NAME)
f.close()