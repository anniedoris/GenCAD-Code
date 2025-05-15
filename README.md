# Real Photo Test Set (WIP)

### Using Dataset from HF
TODO

### Creating HF Dataset

First, load HEICs into the ```real_photo_test_set/heics``` directory. Then, run the following to convert HEICs into PNGs:

```
python scripts/process_heic.py --heic_dir real_photo_test_set/heics --save_dir real_photo_test_set/pngs
```

Next, run the script to generate the HF repo:

TODO