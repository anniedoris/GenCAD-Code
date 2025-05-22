import pandas as pd
import os
from datasets import Dataset, Features, ClassLabel, Value, Image
from huggingface_hub import login
import json
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from prompts import DEFAULT_PROMPT

IMAGE_DIR = 'real_photo_test_set/pngs' # Directory where the pngs are stored
df = pd.read_excel('real_photo_test_set/RealPhotoTestSet.xlsx') # Excel sheet that contains info about photos
deepcad_id = pd.read_excel('real_photo_test_set/id_deepcad_pairs.xlsx', dtype={"DeepCAD_ID": str}) # Excel sheet that contains the mapping from object id to DeepCAD id

print(deepcad_id)

# Map pngs to image info
png_filenames = os.listdir(IMAGE_DIR)
sorted_files = sorted(png_filenames)
df['image'] = [IMAGE_DIR + "/" + i for i in sorted_files]

# Match deepcad ids with object ids
df["Object_ID"] = df["Object_ID"].astype(int)
deepcad_id["Object_ID"] = deepcad_id["Object_ID"].astype(int)
df = df.merge(deepcad_id, on="Object_ID", how="left")
df['question_id'] = df['DeepCAD_ID'] + '_' + (df.groupby('DeepCAD_ID').cumcount() + 1).astype(str)
df['text'] = DEFAULT_PROMPT
df['category'] = 'default'
df['ground_truth'] = '' #TODO: can I remove this? I don't think ground_truth is actually used

# Create question ID
print(df.head())
print(df.columns)
        
# Define the features for HF dataset
features = Features({
    "image": Image(),       # this will load image files
    "Object_ID": Value("int64"),
    "Object_Color": Value("string"),
    "Orientation": Value("string"),
    "Proximity": Value("string"),
    "Background": Value("string"),
    "Lighting": Value("string"),
    "Notes": Value("string"),
    "question_id": Value("string"),
    'text': Value('string'),
    'category': Value('string'),
    'ground_truth': Value('string'),
    'DeepCAD_ID': Value('string')
})

# Convert to dataset
dataset = Dataset.from_pandas(df, features=features)

login()

dataset.push_to_hub("CADCODER/real_photo_test")