import pandas as pd
import matplotlib.pyplot as plt

trees = pd.read_csv('data\\tree_data\\ME_TREE.csv')

picture_date = 2023
date_delta = 3

current_trees = trees[abs(picture_date - trees.INVYR) <= date_delta]

species_freq = current_trees.SPCD.value_counts()
print(species_freq, len(species_freq))