import tensorflow as tf
import numpy as np
from PIL import Image
import pandas as pd
from keras.utils import load_img, img_to_array
import matplotlib.pyplot as plt
import tensorflow_addons as tfa

BASE_PATH = "proc_data\\pieces\\square\\serbia\\"
FILE_NAME = "S2A_MSIL2A_20230613T094041_N0509_R036_T34TCR_20230613T135353\\img3400_4800.png"
MODEL_NAME = "base_model"
FRAME_SIZE = 100
FRAME_DEPTH = 3
MODEL_SPECIES_CODES = [12,316,97,241,375,531,371,129,261,318,746,833,541]

species_names = pd.read_excel("data/tree_data/MasterSpecies.xlsx")

img = load_img(BASE_PATH + FILE_NAME, target_size = (FRAME_SIZE, FRAME_SIZE, FRAME_DEPTH))
img = img_to_array(img)
img = img / 255.

x = np.array([img])
model = tf.keras.models.load_model('models/' + MODEL_NAME, custom_objects= {'f1_score': tfa.metrics.F1Score(num_classes=len(MODEL_SPECIES_CODES))})

pred_labels = model.predict(x)[0]

spec_code_map = {}

for spcd in MODEL_SPECIES_CODES:
    sp_name = species_names.loc[species_names["FIA Code"] == int(spcd), "Common Name"]
    spec_code_map[int(spcd)] = sp_name.values[0]

print("Predicted: ", pred_labels)
print(spec_code_map)

pred_labels_pretty = []

for i in range(len(MODEL_SPECIES_CODES)):
    spcd = int(MODEL_SPECIES_CODES[i])
    if float(pred_labels[i]) >= 0.5:
        pred_labels_pretty.append(spec_code_map[spcd])

print("Predicted Species: ", pred_labels_pretty)

image_fp = np.asarray(Image.open(BASE_PATH + FILE_NAME))
plt.imshow(image_fp)
plt.show()