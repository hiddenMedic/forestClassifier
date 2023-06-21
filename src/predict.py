import tensorflow as tf
import numpy as np
from PIL import Image
import pandas as pd
from keras.utils import load_img, img_to_array
import matplotlib.pyplot as plt
import tensorflow_addons as tfa

BASE_PATH = "proc_data\\pieces\\square\\"
FILE_NAME = "S2B_MSIL2A_20230514T153559_N0509_R111_T19TEK_20230514T195610\img6500_300.png"
MODEL_NAME = "resnet"
FRAME_SIZE = 100
FRAME_DEPTH = 3
TREE_TABLE = "proc_data\\final_data.csv"
MODEL_SPECIES_CODES = [12,316,97,241,375,531,371,129,261,318,746,833,541]

df = pd.read_csv(TREE_TABLE)
species_names = pd.read_excel("data/tree_data/MasterSpecies.xlsx")

img = load_img(BASE_PATH + FILE_NAME, target_size = (FRAME_SIZE, FRAME_SIZE, FRAME_DEPTH))
img = img_to_array(img)
img = img / 255.

x = np.array([img])
model = tf.keras.models.load_model('models/' + MODEL_NAME, custom_objects= {'f1_score': tfa.metrics.F1Score(num_classes=len(MODEL_SPECIES_CODES))})

pred_labels = model.predict(x)[0]
true_labels = df.loc[df["image_name"] == FILE_NAME].values[0][1:]

in_spec_code_map = {}
model_spec_code_map = {}

input_data_species = df.columns[1:]
for spcd in input_data_species:
    sp_name = species_names.loc[species_names["FIA Code"] == int(spcd), "Common Name"]
    in_spec_code_map[int(spcd)] = sp_name.values[0]
    
for spcd in MODEL_SPECIES_CODES:
    sp_name = species_names.loc[species_names["FIA Code"] == int(spcd), "Common Name"]
    model_spec_code_map[int(spcd)] = sp_name.values[0]

print("True: ", true_labels)
print("Predicted: ", pred_labels)

print("Dataset species: ", list(in_spec_code_map.values()))
print("Model species: ", list(model_spec_code_map.values()))
intersecting_species = list(set(in_spec_code_map.values()).intersection(model_spec_code_map.values()))
print("Species in both: ", intersecting_species)

true_labels_pretty = []
pred_labels_pretty = []

for i in range(len(input_data_species)):
    spcd = int(input_data_species[i])
    if float(true_labels[i]) >= 0.5:
        true_labels_pretty.append(in_spec_code_map[spcd])

for i in range(len(MODEL_SPECIES_CODES)):
    spcd = int(MODEL_SPECIES_CODES[i])
    if float(pred_labels[i]) >= 0.5:
        pred_labels_pretty.append(model_spec_code_map[spcd])

print("True Species: ", true_labels_pretty)
print("Predicted Species: ", pred_labels_pretty)

image_fp = np.asarray(Image.open(BASE_PATH + FILE_NAME))
plt.imshow(image_fp)
plt.show()