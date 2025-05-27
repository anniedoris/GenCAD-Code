import pandas as pd
import os
import json
import numpy as np
from datasets import Dataset, Features, ClassLabel, Value, Image, DatasetDict
from huggingface_hub import login
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from prompts import DEFAULT_PROMPT
from tqdm import tqdm

IMAGE_TOKEN_COUNT = 577
PROMPT_TOKEN_COUNT = 69

def read_python_file(filepath):
    with open(filepath, 'r') as file:
        return file.read()

def get_question_ids(jsonl_path):
    with open(jsonl_path, 'r') as f:
        return [json.loads(line)["question_id"] for line in f if "question_id" in json.loads(line)]

# Get the total token counts for each sample. Token counts were generated using the LLaVA repo
df_train = pd.read_json("deepcad_derived/cadquery_train_tokencount.jsonl", lines=True)
df_test = pd.read_json("deepcad_derived/cadquery_test_tokencount.jsonl", lines=True)
merged_df = pd.concat([df_train, df_test], ignore_index=True)
token_count_df = merged_df.set_index('question_id')
full_index = pd.RangeIndex(start=token_count_df.index.min(), stop=token_count_df.index.max() + 1)
token_count_df = token_count_df.reindex(full_index)

total_token_counts = []
for i, row in token_count_df.iterrows():
    if pd.isna(row['ground_truth_token_count']):
        total_token_count = None
    else:
        total_token_count = int(row['ground_truth_token_count'][0]) + IMAGE_TOKEN_COUNT + PROMPT_TOKEN_COUNT
    total_token_counts.append(total_token_count)
    
token_count_df['total_token_count'] = total_token_counts
token_count_df['total_token_count'] = token_count_df['total_token_count'].astype('Int64')

# Load in test train split
df = pd.read_csv('deepcad_derived/split.csv')
df['total_token_count'] = token_count_df['total_token_count']

# Get which samples were in the random 100 test subset
hundred_test = get_question_ids('deepcad_derived/cadquery_test_data_subset100.jsonl')

# Merge everything together
deepcad_ids = []
images = []
cadquery = []
token_count = []
splits = []
hundred_subset = []

for i, row in tqdm(df.iterrows(), total=len(df)):
    id = row['deepcad_id']
    deepcad_ids.append(id)
    images.append(f'deepcad_derived/data/images/{id}_0.png')
    cadquery.append(read_python_file(f'deepcad_derived/data/cadquery/{id}.py'))
    token_count.append(row['total_token_count'])
    splits.append(row['split'])
    if i in hundred_test:
        hundred_subset.append(True)
    else:
        hundred_subset.append(False)
    
hf_df = pd.DataFrame({'deepcad_id': deepcad_ids, 'image': images, 'cadquery': cadquery, 'token_count': token_count, 'split': splits, 'hundred_subset': hundred_subset})
hf_df['prompt'] = DEFAULT_PROMPT
hf_df['token_count'] = hf_df['token_count'].astype('Int64')
hf_df['hundred_subset'] = hundred_subset
      
# Define the features for HF dataset
features = Features({
    "image": Image(),       # this will load image files
    "deepcad_id": Value("string"),
    "cadquery": Value("string"),
    "token_count": Value("int64"),
    "prompt": Value("string"),
    'hundred_subset': Value("bool")
})

# Create a DatasetDict with splits
dataset_dict = DatasetDict({
    split_name: Dataset.from_pandas(
        hf_df[hf_df["split"] == split_name].drop(columns=["split"]).reset_index(drop=True),
        features=features
    )
    for split_name in hf_df["split"].unique() if split_name != "none" #TODO: sort out "none" during image generation
})

login()

dataset_dict.push_to_hub("CADCODER/GenCAD-Code")