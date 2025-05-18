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


# Initialize counters for each split
split_stats = {
    'train': {'images': 0, 'instances': 0},
    'valid': {'images': 0, 'instances': 0},
    'test': {'images': 0, 'instances': 0}
}

for dataset in datasets:
    print(f"\nProcessing dataset: {dataset}")
    dataset_stats = {
        'train': {'images': 0, 'instances': 0},
        'valid': {'images': 0, 'instances': 0},
        'test': {'images': 0, 'instances': 0}
    }
    
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
            
            # Update statistics
            dataset_stats[split]['images'] += 1
            dataset_stats[split]['instances'] += len(new_lines)
            split_stats[split]['images'] += 1
            split_stats[split]['instances'] += len(new_lines)
        
    # Print dataset statistics
    print("\nDataset Statistics:")
    print("-"*50)
    print(f"Total Images: {sum(stats['images'] for stats in split_stats.values()):,}")
    print(f"Total Instances: {sum(stats['instances'] for stats in split_stats.values()):,}")
    print("\nSplit Distribution:")
    print(f"Train: {split_stats['train']['images']:,} images ({split_stats['train']['images']/sum(stats['images'] for stats in split_stats.values())*100:.1f}%)")
    print(f"Valid: {split_stats['valid']['images']:,} images ({split_stats['valid']['images']/sum(stats['images'] for stats in split_stats.values())*100:.1f}%)")
    print(f"Test:  {split_stats['test']['images']:,} images ({split_stats['test']['images']/sum(stats['images'] for stats in split_stats.values())*100:.1f}%)")

# Print final statistics
print("-"*100)
print("\nFinal Dataset Statistics:")
for split in ['train', 'valid', 'test']:
    print(f"\n{split.capitalize()} Split:")
    print(f"  Total Images: {split_stats[split]['images']}")
    print(f"  Total Instances: {split_stats[split]['instances']}")
    print(f"  Average Instances per Image: {split_stats[split]['instances']/split_stats[split]['images']:.2f}")

    """

Train Split:
  Total Images: 3363
  Total Instances: 5695
  Average Instances per Image: 1.69

Valid Split:
  Total Images: 903
  Total Instances: 1472
  Average Instances per Image: 1.63

Test Split:
  Total Images: 401
  Total Instances: 648
  Average Instances per Image: 1.62
  """