import os
import shutil
from glob import glob
from tqdm import tqdm

datasets = [
    'screen-1',
    'screens-segmentation-5',
    'laptop-screen-detection-1',
    'laptop-screen-detection-vivek-1',
]
splits = ['train', 'valid', 'test']
dataset_stats = {split: {'images': 0, 'instances': 0} for split in splits}

for dataset in datasets:
    print(f"\nProcessing dataset: {dataset}")
    split_stats = {split: {'images': 0, 'instances': 0} for split in splits}

    for split in splits:
        print(f"  Processing {split} split...")
        image_input_dir = f'{dataset}/{split}/images'
        label_input_dir = f'{dataset}/{split}/labels'
        image_output_dir = f'segmentation-dataset/{split}/images'
        label_output_dir = f'segmentation-dataset/{split}/labels'

        os.makedirs(image_output_dir, exist_ok=True)
        os.makedirs(label_output_dir, exist_ok=True)

        # Copy and count images
        for img_path in tqdm(glob(f'{image_input_dir}/*'), desc=f"    Copying {split} images"):
            shutil.copy(img_path, os.path.join(image_output_dir, os.path.basename(img_path)))

        # Process and copy label files
        for label_path in tqdm(glob(f'{label_input_dir}/*.txt'), desc=f"    Converting {split} labels"):
            filename = os.path.basename(label_path)
            with open(label_path, 'r') as f:
                lines = f.readlines()

            new_lines = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 3:
                    new_line = '0 ' + ' '.join(parts[1:])  # Replace class ID with 0
                    new_lines.append(new_line)

            with open(os.path.join(label_output_dir, filename), 'w') as f:
                f.write('\n'.join(new_lines))

            split_stats[split]['images'] += 1
            split_stats[split]['instances'] += len(new_lines)
            dataset_stats[split]['images'] += 1
            dataset_stats[split]['instances'] += len(new_lines)

    # Print per-dataset stats
    total_imgs = sum(stats['images'] for stats in split_stats.values())
    print("\nDataset Statistics:")
    print(f"Total Images: {total_imgs:,}")
    print(f"Total Instances: {sum(stats['instances'] for stats in split_stats.values()):,}")
    print("\nSplit Distribution:")
    for split in splits:
        count = split_stats[split]['images']
        percent = (count / total_imgs) * 100 if total_imgs else 0
        print(f"{split.capitalize()}: {count:,} images ({percent:.1f}%)")
    print(split_stats)
    print("-" * 50)

# Final summary
print("-" * 100)
print("\nFinal Dataset Statistics:")
for split in splits:
    total_imgs = dataset_stats[split]['images']
    total_instances = dataset_stats[split]['instances']
    avg = total_instances / total_imgs if total_imgs else 0
    print(f"\n{split.capitalize()} Split:")
    print(f"  Total Images: {total_imgs}")
    print(f"  Total Instances: {total_instances}")
    print(f"  Average Instances per Image: {avg:.2f}")
