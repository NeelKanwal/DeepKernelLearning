""" Authored by: Neel Kanwal (neel.kanwal0@gmail.com)"""

# This file processes TCGAFocus dataset downloaded from https://zenodo.org/records/3910757
# It places artifact free and blur patches in corresponding directories so Inference.py can be run here.

import os
import shutil

# Define the paths to the original and transformed folders
original_folder = "OriginalFolder"
transformed_folder = "TransformedTCGAFocus"

# Create the transformed folder if it doesn't exist
if not os.path.exists(transformed_folder):
    os.makedirs(transformed_folder)

# Create the InFocus and Out-of-Focus folders inside the transformed folder
infocus_folder = os.path.join(transformed_folder, "InFocus")
outoffocus_folder = os.path.join(transformed_folder, "OutofFocus")
os.makedirs(infocus_folder)
os.makedirs(outoffocus_folder)

print("--Starting InFocus--")
# Copy the in-focus files to the InFocus folder
infocus_source = os.path.join(original_folder, "In Focus")
infocus_count = 0
for root, dirs, files in os.walk(infocus_source):
    for file in files:
        if file.endswith(".png"):
            source_path = os.path.join(root, file)
            destination_path = os.path.join(infocus_folder, file)
            shutil.copyfile(source_path, destination_path)
            infocus_count += 1

print(f"Number of files copied to InFocus: {infocus_count}")

print("--Starting OutFocus--")
# Copy the out-of-focus files to the Out-of-Focus folder
outoffocus_source1 = os.path.join(original_folder, "Out of Focus")
outoffocus_source2 = os.path.join(original_folder, "Out of Focus (Marker)")
outoffocus_sources = [outoffocus_source1, outoffocus_source2]
outoffocus_count = 0
for outoffocus_source in outoffocus_sources:
    for root, dirs, files in os.walk(outoffocus_source):
        for file in files:
            if file.endswith(".png"):
                source_path = os.path.join(root, file)
                destination_path = os.path.join(outoffocus_folder, file)
                shutil.copyfile(source_path, destination_path)
                outoffocus_count += 1

# Remove files less than 6KB as they do not have any image data in it.

for filename in os.listdir(infocus_folder):
    filepath = os.path.join(infocus_folder, filename)
    if os.path.isfile(filepath) and filepath.endswith('.png') and os.path.getsize(filepath) < 6 * 1024:
        os.remove(filepath)

for filename in os.listdir(outoffocus_folder):
    filepath = os.path.join(outoffocus_folder, filename)
    if os.path.isfile(filepath) and filepath.endswith('.png') and os.path.getsize(filepath) < 6 * 1024:
        os.remove(filepath)


print(f"Number of files copied to Out-of-Focus: {outoffocus_count}")
print("--Finished--")