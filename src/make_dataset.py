import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import cv2
from tqdm import tqdm

"""
IMAGE_BASE = "data\\image_data\\useful\\"
OUT_FOLDER = "proc_data\\pieces\\square\\"

trees = pd.read_csv('data\\tree_data\\ME_TREE.csv')
plots = pd.read_csv('data\\tree_data\\ME_PLOT.csv')
leader_file = open("proc_data/pieces/square/code inputs.txt", "r")
"""
OUT_FOLDER = "proc_data\\pieces\\square\\alabama\\"
IMAGE_BASE = "data\\image_data\\raw\\_alabama\\S2B_MSIL1C_20230608T161829_N0509_R040_T16REV_20230608T182205\\GRANULE\\a\\IMG_DATA\\"

trees = pd.read_csv('data\\tree_data\\AL_TREE.csv')
plots = pd.read_csv('data\\tree_data\\AL_PLOT.csv')
leader_file = open("proc_data/pieces/square/alabama/code inputs.txt", "r")

leader_file = leader_file.read().split("===")
for batch in leader_file:
    lines = batch.splitlines()
    if lines[0] == '':
        lines = lines[1:]
    
    print(lines)
    PICTURE_DATE = int(lines[1].split("=")[1])
    IMAGE_LAT_TL = float(lines[2].split("=")[1])
    IMAGE_LON_TL = float(lines[3].split("=")[1])
    IMAGE_LAT_BR = float(lines[4].split("=")[1])
    IMAGE_LON_BR = float(lines[5].split("=")[1])
    FOLDER_NAME = lines[6].split("\"")[1]
    IMAGE_NAME = lines[7].split("\"")[1]

    img = cv2.imread(IMAGE_BASE + IMAGE_NAME)
    IMAGE_WIDTH = img.shape[1]
    IMAGE_HEIGHT = img.shape[0]
    FRAME_WIDTH = 100
    FRAME_HEIGHT = 100
    DATE_DELTA = 5
    MIN_SP_FREQ = 1000
    AUGMENTS = 3

    # frame and image HAVE to be square

    print("Total trees: ", len(trees))
    useful_trees = trees[abs(PICTURE_DATE - trees.INVYR) <= DATE_DELTA]
    species_freq = useful_trees.SPCD.value_counts()
    print("Trees with good date: ", len(useful_trees))

    useful_trees = useful_trees[useful_trees.groupby('SPCD')['SPCD'].transform('count').ge(MIN_SP_FREQ)]
    print("Trees with good date and species frequency", len(useful_trees))

    # print(useful_trees)

    result_cols = ['image_name']
    for x, y in species_freq.items():
        if y >= MIN_SP_FREQ:
            result_cols.append(int(x))
        
    print(result_cols)
    result = pd.DataFrame(columns=result_cols)

    lat_height = abs(IMAGE_LAT_TL - IMAGE_LAT_BR)
    lon_width = abs(IMAGE_LON_TL - IMAGE_LON_BR)

    # square needed here
    frame_lat_height = lat_height / (IMAGE_HEIGHT // FRAME_HEIGHT)
    frame_lon_width = lon_width / (IMAGE_WIDTH // FRAME_WIDTH)

    print('rows to iter: ', len(useful_trees))
    for index, row in tqdm(useful_trees.iterrows(), mininterval=10):
        plot_entry = plots.loc[plots['PLOT'] == row['PLOT']].iloc[0]
        lat = plot_entry['LAT']
        lon = plot_entry['LON']
        
        if not(IMAGE_LAT_TL <= lat <= IMAGE_LAT_BR or IMAGE_LAT_TL >= lat >= IMAGE_LAT_BR):
            continue
        if not(IMAGE_LON_TL <= lon <= IMAGE_LON_BR or IMAGE_LON_TL >= lon >= IMAGE_LON_BR): # doesnt pass this
            continue
        
        lat_dist = abs(IMAGE_LAT_TL - lat)
        lon_dist = abs(IMAGE_LON_TL - lon)
        
        frame_height = (lat_dist // frame_lat_height) * FRAME_HEIGHT # this is not accurate since lat and lon are curved, not grid-like
        frame_width = (lon_dist // frame_lon_width) * FRAME_WIDTH # but its an ok approximation
        frame_height = int(frame_height)
        frame_width = int(frame_width)
        
        image_names = [FOLDER_NAME + f"\\img{frame_width}_{frame_height}.png"]
        for i in range(AUGMENTS):
            image_names.append(FOLDER_NAME + f"\\img{frame_width}_{frame_height}_{i}.png")
        
        already_in = (result['image_name'] == image_names[0]).any()
        # print(image_names, image_names[0], already_in)
        
        if not already_in:
            for name in image_names:
                result = pd.concat([pd.DataFrame([[name] + [0] * (len(result.columns) - 1)], columns=result.columns), result], ignore_index=True)
            
        for name in image_names:
            result.loc[result['image_name'] == name, row['SPCD']] += 1


    result.to_csv(OUT_FOLDER + FOLDER_NAME + '.csv', index=False)
    print('written to csv.')