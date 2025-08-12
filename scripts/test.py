import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd

df = pd.read_csv('deepcad_derived/gencad_split.csv')
print(df)

deepcad_ids = []
split = []
for i, row in df.iterrows():
    orig_id = row['image_path'].split('_0.png')[0].split('images/')[-1]
    if row['split'] == 'valiation':
        split_value = "validation"
    else:
        split_value = row['split']
    deepcad_ids.append(orig_id)
    split.append(split_value)
        
        
df_final = pd.DataFrame({'deepcad_id': deepcad_ids, 'split': split})
df_final.to_csv('deepcad_derived/split.csv')