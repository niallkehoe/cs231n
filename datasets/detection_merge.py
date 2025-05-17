"""
We have a bunch of detection datasets, and we want to merge them into a single dataset.
Copy over data by run the following command:

# -p for create parent directories if they don't exist
mkdir -p detection-dataset/train/images
mkdir -p detection-dataset/valid/images
mkdir -p detection-dataset/test/images
mkdir -p detection-dataset/train/labels
mkdir -p detection-dataset/valid/labels
mkdir -p detection-dataset/test/labels

# Copy images
cp Computer-monitor-1/train/images/* detection-dataset/train/images/
cp Computer-monitor-1/valid/images/* detection-dataset/valid/images/
cp Computer-monitor-1/test/images/* detection-dataset/test/images/
cp Monitors\(Tvs,Pc-Monitors,-etc\)-2/train/images/* detection-dataset/train/images/
cp Monitors\(Tvs,Pc-Monitors,-etc\)-2/valid/images/* detection-dataset/valid/images/
cp Monitors\(Tvs,Pc-Monitors,-etc\)-2/test/images/* detection-dataset/test/images/
cp Office-Monitor-1/train/images/* detection-dataset/train/images/
cp Office-Monitor-1/valid/images/* detection-dataset/valid/images/
cp Office-Monitor-1/test/images/* detection-dataset/test/images/
cp screen-1/train/images/* detection-dataset/train/images/
cp screen-1/valid/images/* detection-dataset/valid/images/
cp screen-1/test/images/* detection-dataset/test/images/

# Copy labels
cp Computer-monitor-1/train/labels/* detection-dataset/train/labels/
cp Computer-monitor-1/valid/labels/* detection-dataset/valid/labels/
cp Computer-monitor-1/test/labels/* detection-dataset/test/labels/
cp Monitors\(Tvs,Pc-Monitors,-etc\)-2/train/labels/* detection-dataset/train/labels/
cp Monitors\(Tvs,Pc-Monitors,-etc\)-2/valid/labels/* detection-dataset/valid/labels/
cp Monitors\(Tvs,Pc-Monitors,-etc\)-2/test/labels/* detection-dataset/test/labels/
cp Office-Monitor-1/train/labels/* detection-dataset/train/labels/
cp Office-Monitor-1/valid/labels/* detection-dataset/valid/labels/
cp Office-Monitor-1/test/labels/* detection-dataset/test/labels/
cp screen-1/train/labels/* detection-dataset/train/labels/
cp screen-1/valid/labels/* detection-dataset/valid/labels/
cp screen-1/test/labels/* detection-dataset/test/labels/
"""

import os
from glob import glob
from tqdm import tqdm

datasets = [
    'Computer-monitor-1',
    'Monitors(Tvs,Pc-Monitors,-etc)-2',
    'Office-Monitor-1',
    'screen-1',
]


for dataset in datasets:
    print(f"\nProcessing dataset: {dataset}")
    for split in ['train', 'valid', 'test']:
        print(f"  Processing {split} split...")
        input_dir = f'{dataset}/{split}/labels'
        output_dir = f'detection-dataset/{split}/labels'
        os.makedirs(output_dir, exist_ok=True)

        for file_path in tqdm(glob(f'{input_dir}/*.txt'), desc=f"    Converting {split} labels"):
            filename = os.path.basename(file_path)
            with open(file_path, 'r') as f:
                lines = f.readlines()

            # Convert all class IDs to 0 [now only have one class]
            new_lines = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 3:
                    new_line = '0 ' + ' '.join(parts[1:])  # Replace class ID with 0
                    new_lines.append(new_line)

            with open(f'{output_dir}/{filename}', 'w') as f:
                f.write('\n'.join(new_lines))