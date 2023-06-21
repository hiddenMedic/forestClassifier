from PIL import Image, ImageOps
import random
import os
from tqdm import tqdm

"""
BASE_PATH = "proc_data\\pieces\\square\\"
folder_names = [
    "S2A_MSIL2A_20211215T153641_N0301_R111_T19TEL_20211215T182207",
    "S2A_MSIL2A_20230519T153601_N0509_R111_T19TDJ_20230519T215600",
    "S2A_MSIL2A_20230522T153811_N0509_R011_T19TCK_20230522T231356",
    "S2A_MSIL2A_20230522T153811_N0509_R011_T19TCL_20230522T231356",
    "S2A_MSIL2A_20230522T153811_N0509_R011_T19TDN_20230522T231356",
    "S2B_MSIL2A_20230514T153559_N0509_R111_T19TDJ_20230514T195610",
    "S2B_MSIL2A_20230514T153559_N0509_R111_T19TDK_20230514T195610",
    "S2B_MSIL2A_20230514T153559_N0509_R111_T19TDL_20230514T195610",
    "S2B_MSIL2A_20230514T153559_N0509_R111_T19TEK_20230514T195610"
]
"""
BASE_PATH = "proc_data\\pieces\\square\\mississippi\\"
FOLDER_NAMES = [
    "S2B_MSIL1C_20230608T161829_N0509_R040_T16REV_20230608T182205"
]

TRANS_PER_FRAME = 3

transformations = [
    "12", "21", "22", "31", "32", "41", "42"
] # "11" is nothing

for folder in tqdm(FOLDER_NAMES):
    lst = os.listdir(BASE_PATH + folder)
    for entry in tqdm(lst, mininterval=10):
        full_name = BASE_PATH + folder + "\\" + entry
        full_name_no_ext = BASE_PATH + folder + "\\" + entry.split('.')[0] # if file has . in name, this crashes
        ext = entry.split('.')[1]
        if not os.path.isfile(full_name) or ext == 'txt' or entry.count("_") > 1:
            continue
        # print(full_name)
        transs = random.sample(transformations, TRANS_PER_FRAME)
        for i in range(TRANS_PER_FRAME):
            trans = transs[i]
            image = Image.open(full_name)
            rot = int(trans[0]) - 1
            flip = int(trans[1]) - 1
            image = image.rotate(rot * 90)
            if flip:
                image = ImageOps.mirror(image)
            image.save(full_name_no_ext + f"_{i}." + ext)