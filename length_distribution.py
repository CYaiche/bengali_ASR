

import json , os 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from common.common_params import EXTRACT_DIR

metadata_folder     = os.path.join(EXTRACT_DIR,"metadata") 
text_metadata       = os.path.join(metadata_folder, "life_clean.csv")
metadata            = pd.read_csv(text_metadata)

length = metadata["sentence"].apply(len)

print(length)
nb = len(length)
x = [i for i  in range(nb)]

q1 = np.quantile(length, 0.25)
q3 = np.quantile(length, 0.75)

plt.scatter(x,length)
plt.axhline(y =q1, color = 'r', linestyle = '-' ,label = "Q1") 
plt.axhline(y = q3, color = 'r', linestyle = '-', label="Q3") 
plt.title("Label length distribution")
plt.show()

print(f"max : {max(length)}")
print(f"min : {min(length)}")
print(f"mean : {np.mean(length)}")