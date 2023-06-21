import pandas as pd
import matplotlib.pyplot as plt

plots = pd.read_csv('data\\tree_data\\ME_PLOT.csv')

LAT_freq = plots.LAT.value_counts()
LON_freq = plots.LON.value_counts()
print(LAT_freq, LON_freq, len(LAT_freq), len(LON_freq))

# Maine centroid is 45°15′11.9982″N 69°13′59.9988″W
# US centroid is 44°58′2.07622″N 103°46′17.60283″W
# neither of these seem to be prominent in the database. GOOD!