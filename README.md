## Summary of Datasets

Our datasets can be accessed using Huggingface or the Downloadable links below.

```CADCODER/GenCAD-Code```: A dataset of XX images of CAD models pair with CadQuery Python code. This dataset is dervied from the DeepCAD dataset.

&nbsp;&nbsp;&nbsp;&nbsp;[Huggingface]() | [Download]()

```CADCODER/real_photo_test```: A dataset of 400 images of 3D printed CAD objects from the test subset of the DeepCAD dataset.

&nbsp;&nbsp;&nbsp;&nbsp;[Huggingface](https://huggingface.co/datasets/CADCODER/real_photo_test) | [Download]()

## Generating our Datasets

We provide the scripts we used to create our datasets below, in case you'd like to adapt them to your own needs.

#### Generating GenCAD-Code:
1. Download the DeepCAD vectors from [this link](https://drive.google.com/drive/folders/1DJU4aqNTbGMnT8NKP9gyWrUoHgHWOCDH?usp=sharing) in the ```deepcad_derived``` directory and unzip.
2. Create the necessary environment using:

```

```

3. Run the following to convert the .h5 vector files into Python CadQuery files. It should take ~2 minutes for all files to generate.

```
python scripts/h5tocadquery.py
```

#### Generating Real Photo Test Data:
First, load HEICs into the ```real_photo_test_set/heics``` directory. Then, run the following to convert HEICs into PNGs:

```
python scripts/process_heic.py --heic_dir real_photo_test_set/heics --save_dir real_photo_test_set/pngs
```

Next, run the script to push the dataset to huggingface:

```
python scripts/upload_realphoto_to_hf.py
```