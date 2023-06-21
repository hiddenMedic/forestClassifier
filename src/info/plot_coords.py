import pandas as pd
import matplotlib.pyplot as plt

plots = pd.read_csv('data\\tree_data\\AL_PLOT.csv')

PICTURE_DATE = 2023
DATE_DELTA = 5

current_plots = plots[abs(PICTURE_DATE - plots.INVYR) <= DATE_DELTA]

LAT_vals = current_plots.LAT.values
LON_vals = current_plots.LON.values

assert(len(LAT_vals) == len(LON_vals))

print(type(LAT_vals), len(LAT_vals)) # delta = 3: 1442, delta = 5: 2876

plt.scatter(LON_vals, LAT_vals)
plt.show()