"""
We have a bunch of segmentation datasets, and we want to merge them into a single dataset.
Copy over data by run the following command:

# -p for create parent directories if they don't exist
mkdir -p segmentation-dataset/train/images
mkdir -p segmentation-dataset/valid/images
mkdir -p segmentation-dataset/test/images
mkdir -p segmentation-dataset/train/labels
mkdir -p segmentation-dataset/valid/labels
mkdir -p segmentation-dataset/test/labels

# Copy images
cp screen-1/train/images/* segmentation-dataset/train/images/
cp screen-1/valid/images/* segmentation-dataset/valid/images/
cp screen-1/test/images/* segmentation-dataset/test/images/
cp screens-segmentation-5/train/images/* segmentation-dataset/train/images/
cp screens-segmentation-5/valid/images/* segmentation-dataset/valid/images/
cp screens-segmentation-5/test/images/* segmentation-dataset/test/images/

# Copy labels
cp screen-1/train/labels/* segmentation-dataset/train/labels/
cp screen-1/valid/labels/* segmentation-dataset/valid/labels/
cp screen-1/test/labels/* segmentation-dataset/test/labels/
cp screens-segmentation-5/train/labels/* segmentation-dataset/train/labels/
cp screens-segmentation-5/valid/labels/* segmentation-dataset/valid/labels/
cp screens-segmentation-5/test/labels/* segmentation-dataset/test/labels/
"""

import os
from glob import glob
from tqdm import tqdm

datasets = [
    'screen-1',
    'screens-segmentation-5',
]

dataset_stats = {
    'train': {'images': 0, 'instances': 0},
    'valid': {'images': 0, 'instances': 0},
    'test': {'images': 0, 'instances': 0}
}

for dataset in datasets:
    print(f"\nProcessing dataset: {dataset}")
    # Initialize counters for each split
    split_stats = {
        'train': {'images': 0, 'instances': 0},
        'valid': {'images': 0, 'instances': 0},
        'test': {'images': 0, 'instances': 0}
    }
    
    for split in ['train', 'valid', 'test']:
        print(f"  Processing {split} split...")
        input_dir = f'{dataset}/{split}/labels'
        output_dir = f'segmentation-dataset/{split}/labels'
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
            split_stats[split]['images'] += 1
            split_stats[split]['instances'] += len(new_lines)
            dataset_stats[split]['images'] += 1
            dataset_stats[split]['instances'] += len(new_lines)
        
    # Print dataset statistics
    print("\nDataset Statistics:")
    print(f"Total Images: {sum(stats['images'] for stats in split_stats.values()):,}")
    print(f"Total Instances: {sum(stats['instances'] for stats in split_stats.values()):,}")
    print("\nSplit Distribution:")
    print(f"Train: {split_stats['train']['images']:,} images ({split_stats['train']['images']/sum(stats['images'] for stats in split_stats.values())*100:.1f}%)")
    print(f"Valid: {split_stats['valid']['images']:,} images ({split_stats['valid']['images']/sum(stats['images'] for stats in split_stats.values())*100:.1f}%)")
    print(f"Test:  {split_stats['test']['images']:,} images ({split_stats['test']['images']/sum(stats['images'] for stats in split_stats.values())*100:.1f}%)")
    print(split_stats)
    print("-"*50)

# Print final statistics
print("-"*100)
print("\nFinal Dataset Statistics:")
for split in ['train', 'valid', 'test']:
    print(f"\n{split.capitalize()} Split:")
    print(f"  Total Images: {dataset_stats[split]['images']}")
    print(f"  Total Instances: {dataset_stats[split]['instances']}")
    print(f"  Average Instances per Image: {dataset_stats[split]['instances']/dataset_stats[split]['images']:.2f}")

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