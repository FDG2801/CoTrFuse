import os
import pandas as pd

# Get all files in the current directory
all_files = os.listdir()

# Filter for .jpg files
image_files = [f for f in all_files if f.endswith('.png')]

# Prepare data for CSV
rows = []
for image_file in image_files:
    rows.append((image_file))

# Create DataFrame and write to CSV
df = pd.DataFrame(rows, columns=['image_name'])
df.to_csv('val_lungsegdata_complete.csv', index=False)