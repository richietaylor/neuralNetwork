# Usage:

# python compress_images_proportional.py --input_dir images/train --output_dir images/train_compressed --scale_factor 0.5 --quality 85



import os
import argparse
from PIL import Image
from pathlib import Path
from tqdm import tqdm

def compress_images_proportional(input_dir, output_dir, scale_factor=0.5, quality=85):
    """
    Compresses images by resizing them proportionally based on a scaling factor.

    Parameters:
    - input_dir (str): Path to the input directory containing images.
    - output_dir (str): Path to the output directory to save compressed images.
    - scale_factor (float): Scaling factor for resizing images (e.g., 0.5 reduces size by 50%).
    - quality (int): Quality of saved images (1-95). Higher means better quality.

    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Supported image extensions
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')

    # Get list of image files
    image_files = [f for f in input_dir.iterdir() if f.suffix.lower() in image_extensions]

    print(f"Compressing {len(image_files)} images from '{input_dir}' to '{output_dir}'...")

    for image_file in tqdm(image_files, desc="Compressing Images"):
        try:
            with Image.open(image_file) as img:
                # Convert to RGB if necessary
                if img.mode in ("RGBA", "P"):
                    img = img.convert("RGB")

                # Get original dimensions
                orig_width, orig_height = img.size

                # Calculate new dimensions
                new_width = int(orig_width * scale_factor)
                new_height = int(orig_height * scale_factor)

                # Ensure dimensions are at least 1 pixel
                new_width = max(1, new_width)
                new_height = max(1, new_height)

                # Resize image while maintaining aspect ratio
                img_resized = img.resize((new_width, new_height), Image.LANCZOS)

                # Save compressed image
                output_file = output_dir / image_file.name
                img_resized.save(output_file, optimize=True, quality=quality)

        except Exception as e:
            print(f"Error processing {image_file}: {e}")

    print("Image compression completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compress images proportionally by resizing based on a scaling factor.")
    parser.add_argument('--input_dir', type=str, required=True, help="Path to the input directory containing images.")
    parser.add_argument('--output_dir', type=str, required=True, help="Path to the output directory to save compressed images.")
    parser.add_argument('--scale_factor', type=float, default=0.5, help="Scaling factor for resizing images (e.g., 0.5 reduces size by 50%).")
    parser.add_argument('--quality', type=int, default=85, help="Quality of saved images (1-95). Higher means better quality.")

    args = parser.parse_args()

    compress_images_proportional(args.input_dir, args.output_dir, scale_factor=args.scale_factor, quality=args.quality)
