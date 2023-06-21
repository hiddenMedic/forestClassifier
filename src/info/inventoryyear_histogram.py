import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('data/tree_data/ME_TREE.csv')
print(data.head())
hist = data.hist(column="INVYR", bins=27, rwidth=0.9)
for x in hist[0]: 
    print(x)

plt.show()