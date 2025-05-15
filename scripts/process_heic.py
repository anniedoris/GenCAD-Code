import os
import argparse
import pillow_heif
from PIL import Image

def process_heic(image_path, output_path):
    # Load HEIC image
    heif_file = pillow_heif.open_heif(image_path)
    image = Image.frombytes(
        heif_file.mode, 
        heif_file.size, 
        heif_file.data, 
        "raw", 
        heif_file.mode
    ) 
    
    # Get original dimensions
    width, height = image.size
    
    # Crop to square (center crop)
    min_side = min(width, height)
    left = (width - min_side) // 2
    top = (height - min_side) // 2
    right = left + min_side
    bottom = top + min_side
    image = image.crop((left, top, right, bottom))
    
    # Resize if necessary
    if min_side > 448:
        image = image.resize((448, 448), Image.LANCZOS)
    
    # Save as PNG
    print(image.size)
    image.save(output_path, "PNG")
    print(f"Saved processed image to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process HEIC images to square PNGs.")
    parser.add_argument("--heic_dir", required=True, help="Directory containing HEIC files.")
    parser.add_argument("--save_dir", required=True, help="Directory to save pngs.")
    args = parser.parse_args()

    heic_dir = args.heic_dir
    heic_files = [f for f in os.listdir(heic_dir) if f.lower().endswith('.heic')]

    for heic in heic_files:
        im = os.path.join(heic_dir, heic)
        im_save_png = im[:-5] + ".png"
        im_save_png = im_save_png.replace(args.heic_dir, args.save_dir)
        
        # Ensure the output directory exists
        os.makedirs(os.path.dirname(im_save_png), exist_ok=True)
        
        # Check that it already exists
        if not os.path.exists(im_save_png):
            process_heic(im, im_save_png)