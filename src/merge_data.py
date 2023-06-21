import pandas as pd

base_path = "proc_data\\pieces\\square\\"
file_names = [
    "S2A_MSIL2A_20211215T153641_N0301_R111_T19TEL_20211215T182207.csv",
    "S2A_MSIL2A_20230519T153601_N0509_R111_T19TDJ_20230519T215600.csv",
    "S2A_MSIL2A_20230522T153811_N0509_R011_T19TCK_20230522T231356.csv",
    "S2A_MSIL2A_20230522T153811_N0509_R011_T19TCL_20230522T231356.csv",
    "S2A_MSIL2A_20230522T153811_N0509_R011_T19TDN_20230522T231356.csv",
    "S2B_MSIL2A_20230514T153559_N0509_R111_T19TDJ_20230514T195610.csv",
    "S2B_MSIL2A_20230514T153559_N0509_R111_T19TDK_20230514T195610.csv",
    "S2B_MSIL2A_20230514T153559_N0509_R111_T19TDL_20230514T195610.csv",
    "S2B_MSIL2A_20230514T153559_N0509_R111_T19TEK_20230514T195610.csv"
]

total_data = pd.read_csv(base_path + file_names[0])

for i in range(1, len(file_names)):
    data = pd.read_csv(base_path + file_names[i])
    total_data = pd.concat([total_data, data], ignore_index=True)
    
total_data.fillna(0, inplace=True)

total_data.to_csv("proc_data\\merged_data.csv", index=False)
print("Done")