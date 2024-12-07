import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import pandas as pd
import matplotlib.pyplot as plt
import textwrap
import random
import pickle
from functools import lru_cache
from torchvision import transforms
from utils import clean_and_validate_attributes, generate_natural_description, remove_punc_special

def cache_dataset(dataset, cache_file):
    """Cache dataset to disk"""
    with open(cache_file, 'wb') as f:
        pickle.dump(dataset, f)

def load_cached_dataset(cache_file):
    """Load dataset from cache"""
    with open(cache_file, 'rb') as f:
        return pickle.load(f)

def load_pose_annotations(root_dir):
    pose_file = os.path.join(root_dir, 'CelebAMask-HQ-pose-anno.txt')
    pose_data = {}

    with open(pose_file, 'r') as f:
        # Skip first two lines (count and header)
        next(f)
        next(f)

        for line in f:
            parts = line.strip().split()
            if len(parts) == 4: # image_name, yaw, pitch, roll
                img_id = parts[0].replace('.jpg', '')
                pose_data[img_id] = {
                    'yaw': float(parts[1]),
                    'pitch': float(parts[2]),
                    'roll': float(parts[3])
                }

    return pose_data

def is_good_pose(pose, thresholds):
    return (abs(pose['yaw']) <= thresholds['yaw'] and
            abs(pose['pitch']) <= thresholds['pitch'] and
            abs(pose['roll']) <= thresholds['roll'])

def load_celeba_dataset_with_annotations(root_dir):
    POSE_THRESHOLDS = {
        'yaw': 20.0,
        'pitch': 15.0,
        'roll': 15.0
    }

    pose_data = load_pose_annotations(root_dir)

    SELECTED_ATTRIBUTES = [
        'Young', 'Male',
        'Smiling', 'Eyeglasses',
        'Black_Hair', 'Blond_Hair', 'Bald',
        'Mustache', 'Wearing_Lipstick'
    ]

    attr_file = os.path.join(root_dir, 'CelebAMask-HQ-attribute-anno.txt')
    with open(attr_file, 'r') as f:
        next(f)  # Skip first line (number of images)
        attr_names = next(f).strip().split()
        attr_data = []
        for line in f:
            parts = line.strip().split()
            if len(parts) > 0:
                values = [int(x) for x in parts[1:]]
                attr_data.append(values)

    attr_df = pd.DataFrame(attr_data, columns=attr_names)
    attr_df = (attr_df + 1) // 2  # Convert -1/1 to 0/1

    dataset = []
    filtered_stats = {
        'total': 0,
        'pose_filtered': 0,
        'processed': 0
    }

    for idx in range(len(attr_df)):
        img_id = str(idx)
        filtered_stats['total'] += 1

        if img_id not in pose_data or not is_good_pose(pose_data[img_id], POSE_THRESHOLDS):
            filtered_stats['pose_filtered'] += 1
            continue

        img_path = os.path.join(root_dir, 'CelebA-HQ-img', f'{idx}.jpg')
        if not os.path.exists(img_path):
            continue

        image = Image.open(img_path).convert('RGB')
        image = image.resize((64, 64), Image.Resampling.LANCZOS)
        image = np.array(image)

        attrs = []
        for attr in SELECTED_ATTRIBUTES:
            if attr_df.loc[idx, attr] == 1:
                attr_text = attr.replace('_', ' ').lower()
                attrs.append(attr_text)

        # Infer 'female' if 'male' not present
        if 'male' not in attrs:
            attrs.append('female')

        if len(attrs) < 2:
            continue

        description = generate_natural_description(attrs)

        dataset.append({
            'img_id': img_id,
            'image': image,
            'natural_caption': description,
            'attributes': attrs,
            'pose': pose_data[img_id]
        })
        filtered_stats['processed'] += 1

    print("\nDataset Filtering Statistics:")
    print(f"Total images: {filtered_stats['total']}")
    print(f"Filtered due to pose: {filtered_stats['pose_filtered']} ({filtered_stats['pose_filtered']/filtered_stats['total']*100:.1f}%)")
    print(f"Final processed images: {filtered_stats['processed']} ({filtered_stats['processed']/filtered_stats['total']*100:.1f}%)")

    return dataset, attr_df

def filter_by_resolution_and_sharpness(dataset, min_resolution=(64,64)):
    filtered_dataset = []
    for entry in dataset:
        image = Image.fromarray(entry['image'])
        if image.size >= min_resolution:
            filtered_dataset.append(entry)
    return filtered_dataset

@lru_cache(maxsize=None)
def get_celeba_dataset(root_dir, cache_path='celeba_dataset_cache.pkl'):
    """Get CelebA-HQ dataset with caching"""
    if os.path.exists(cache_path):
        print("Loading dataset from cache...")
        return load_cached_dataset(cache_path)

    print("Processing CelebA-HQ dataset from scratch...")
    dataset, attributes_df = load_celeba_dataset_with_annotations(root_dir)

    filtered_dataset = filter_by_resolution_and_sharpness(dataset, min_resolution=(64,64))

    cache_dataset(filtered_dataset, cache_path)
    print("Final dataset size:", len(filtered_dataset))
    return filtered_dataset

def get_celeba_subset(root_dir, subset_size=5000, random_subset=True, cache_path='celeba_subset_cache.pkl', seed=42):
    method = 'random' if random_subset else 'sequential'
    balance_type = 'gender_balanced'
    subset_cache_path = cache_path.replace('.pkl', f'_{method}_{balance_type}_{subset_size}.pkl')

    if os.path.exists(subset_cache_path):
        print(f"Loading {method} gender-balanced subset of {subset_size} samples from cache...")
        return load_cached_dataset(subset_cache_path)

    print(f"Processing CelebA-HQ {method} gender-balanced subset of {subset_size} samples from scratch...")

    full_dataset = get_celeba_dataset(root_dir, cache_path)

    if subset_size > len(full_dataset):
        print(f"Warning: Requested subset size {subset_size} is larger than dataset size {len(full_dataset)}")
        subset_size = len(full_dataset)

    min_count = 3
    def has_minimum_attributes(item, min_count):
        attributes = item.get('attributes', [])
        return len(set(attributes)) >= min_count

    filtered_dataset = [item for item in full_dataset if has_minimum_attributes(item, min_count)]
    print(f"Dataset filtered to {len(filtered_dataset)} examples with at least {min_count} attributes.")

    male_examples, female_examples = [], []
    for item in filtered_dataset:
        if 'male' in item['attributes']:
            male_examples.append(item)
        else:
            female_examples.append(item)

    print(f"Total male examples: {len(male_examples)}")
    print(f"Total female examples: {len(female_examples)}")

    target_size = subset_size // 2
    random.seed(seed)

    if random_subset:
        selected_males = random.sample(male_examples, min(target_size, len(male_examples)))
        selected_females = random.sample(female_examples, min(target_size, len(female_examples)))
    else:
        selected_males = male_examples[:target_size]
        selected_females = female_examples[:target_size]

    balanced_subset = selected_males + selected_females
    if random_subset:
        random.shuffle(balanced_subset)

    print(f"\nFinal gender-balanced subset: {len(balanced_subset)} (Male: {len(selected_males)}, Female: {len(selected_females)})")

    cache_dataset(balanced_subset, subset_cache_path)
    print(f"Subset cached at: {subset_cache_path}")
    return balanced_subset

def custom_collate_fn(batch):
    images = torch.stack([item['image'] for item in batch])
    attributes = torch.stack([item['attributes'] for item in batch])
    captions = [item['caption'] for item in batch]
    poses = [item['pose'] for item in batch]

    batch_dict = {
        'image': images,
        'attributes': attributes,
        'caption': captions,
        'pose': poses
    }
    return batch_dict

class CelebADataset(Dataset):
    def __init__(self, dataset, max_length=64):
        self.dataset = dataset
        self.max_length = max_length

        self.attribute_to_idx = {
            'young': 0,
            'male': 1,
            'female': 2,
            'smiling': 3,
            'eyeglasses': 4,
            'black_hair': 5,
            'blond_hair': 6,
            'bald': 7,
            'mustache': 8,
            'wearing_lipstick': 9
        }

        self.num_attributes = len(self.attribute_to_idx)

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = Image.fromarray(item['image'])
        image = self.transform(image)
        caption_text = item['natural_caption']

        attributes = torch.zeros(self.num_attributes)
        for attr in item['attributes']:
            attr_key = attr.lower().replace(' ','_')
            if attr_key in self.attribute_to_idx:
                attributes[self.attribute_to_idx[attr_key]] = 1.0

        return {
            'image': image,
            'attributes': attributes,
            'caption': caption_text,
            'pose': item['pose']
        }

def visualize_dataset_samples(dataset, num_samples=5, save_path='dataset_samples.png'):    
    print("dataset size:", len(dataset))
    fig, axes = plt.subplots(num_samples, 1, figsize=(10, 4*num_samples))
    indices = random.sample(range(len(dataset)), num_samples)

    for idx, ax in zip(indices, axes):
        sample = dataset[idx]
        image = sample['image']
        caption = sample['caption']

        image = (image.permute(1,2,0) + 1) / 2
        image = image.numpy()

        ax.imshow(image)
        ax.axis('off')
        wrapped_text = textwrap.fill(caption, width=60)
        ax.set_title(wrapped_text, pad=10)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()
