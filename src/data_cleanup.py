import pandas as pd

INPUT_FILE = "proc_data\\pieces\\square\\alabama\\S2B_MSIL1C_20230608T161829_N0509_R040_T16REV_20230608T182205.csv"
OUTPUT_FILE = "proc_data\\pieces\\square\\alabama\\final_data_al.csv"

df = pd.read_csv(INPUT_FILE)

MIN_NEEDED_TREES = 7 # i need this many trees for it to be counted as having that tree type
MAX_COLUMN_ZERO_PCT = 95 # if more than 95% of instances arent assigned this class, delete this class

col_to_drop = []

for column in df.columns:
    if column == "image_name":
        continue
    df.loc[df[column] < MIN_NEEDED_TREES, column] = 0
    df.loc[df[column] != 0, column] = 1
    
    count = (df[column] == 0).sum()
    pct = round(count / len(df.index) * 100, 2)
    print(f"{column} has {count} / {len(df.index)} ({pct} %) zeroes." + (f" Deleting this class. ({pct} > {MAX_COLUMN_ZERO_PCT})" if pct > MAX_COLUMN_ZERO_PCT else ""))
    if pct > MAX_COLUMN_ZERO_PCT:
        col_to_drop.append(column) 

print(f"Dropping columns: {col_to_drop}")
df.drop(columns=col_to_drop, inplace=True)

print(f"Dataframe size before deleting zero rows: {len(df.index)}")
df = df[df.drop('image_name', axis=1).ne(0).any(axis=1)].reset_index(drop=True)
print(f"Dataframe size after deleting zero rows: {len(df.index)}")

df.to_csv(OUTPUT_FILE, index=False)

print("\n")
print(df.head())

