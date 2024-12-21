import torch
import os
import numpy as np
import random
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import torch.nn.functional as F

from textwrap import wrap
import textwrap

from src.config import config
from src.data import CelebADataset, get_celeba_subset
from src.models import MultimodalVAE
from src.data.utils import clean_and_validate_attributes
from src.visualization import VisualizationUtils

def load_model_and_data(device, latent_dim, model_checkpoint='final_model.pt'):
    checkpoint = torch.load(model_checkpoint, map_location=device)
    model = MultimodalVAE(
        latent_dim=latent_dim,
        temperature=1.0
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()

    test_dataset = get_celeba_subset(
        root_dir=config['data_path'],
        subset_size=5000,
        random_subset=True,
        cache_path=config['cache_path']
    )
    dataset = CelebADataset(test_dataset)
    return model, dataset
    
def visualize_random_examples(dataset, num_samples=5, save_dir='eval_results'):
    os.makedirs(save_dir, exist_ok=True)
    np.random.seed(None)
    
    total_samples = len(dataset)
    random_indices = np.random.choice(total_samples, num_samples, replace=False)
    
    fig, axes = plt.subplots(1, num_samples, figsize=(3*num_samples, 4))
    
    if num_samples == 1:
        axes = [axes]
    
    for idx, ax in enumerate(axes):
        sample = dataset[random_indices[idx]]
        image = sample['image']
        caption = sample['caption']
        
        show_img = (image + 1) / 2
        ax.imshow(show_img.permute(1,2,0).numpy())
        ax.axis('off')
        
        wrapped_caption = '\n'.join(wrap(caption, width=22.5))
        ax.text(0.5, -0.1, wrapped_caption,
               ha='center', va='top',
               transform=ax.transAxes,
               wrap=True,
               fontsize=18,
               bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.3)
    plt.savefig(os.path.join(save_dir, 'random_dataset_examples.png'),
                bbox_inches='tight',
                dpi=300)
    plt.close(fig)
    
def evaluate_consistency(model, dataset, device, num_samples=100, threshold=0.20):
    model.eval()
    matching_scores = []
    mismatched_scores = []
    correct_classifications = 0
    total_pairs = 0

    MIN_MISMATCH_ATTRS = 4

    # Groups of attributes to ensure diversity in mismatches
    group1 = [1, 2]       # male, female
    group2 = [3, 4, 9]    # smiling, eyeglasses, wearing_lipstick
    group3 = [5, 6, 7]    # black_hair, blond_hair, bald
    group4 = [8]          # mustache
    groups = [group1, group2, group3, group4]

    with torch.no_grad():
        for i in range(num_samples):
            sample = dataset[i]
            image = sample['image'].unsqueeze(0).to(device)
            attributes = sample['attributes'].unsqueeze(0).to(device)

            original_attrs_np = attributes.cpu().numpy().copy()
            original_male = original_attrs_np[0, 1]
            original_female = original_attrs_np[0, 2]

            # Select a mismatched sample (just take the next one in a cyclic manner)
            mismatch_idx = (i + 1) % len(dataset)
            mismatched_sample = dataset[mismatch_idx]
            mismatched_attrs = mismatched_sample['attributes'].unsqueeze(0).to(device)

            diff_count = (mismatched_attrs != attributes).sum().item()

            # Always force gender mismatch first
            mismatch_attrs_np = mismatched_attrs.cpu().numpy().copy()

            # If original was male=1,female=0 => now female=1,male=0
            # If original was female=1,male=0 => now male=1,female=0
            if original_male == 1 and original_female == 0:
                # Originally male, flip to female
                mismatch_attrs_np[0, 1] = 0  # male
                mismatch_attrs_np[0, 2] = 1  # female
            elif original_female == 1 and original_male == 0:
                # Originally female, flip to male
                mismatch_attrs_np[0, 2] = 0  # female
                mismatch_attrs_np[0, 1] = 1  # male
            else:
                # If ambiguous, just set to male by default
                mismatch_attrs_np[0, 1] = 1
                mismatch_attrs_np[0, 2] = 0

            # Recompute diff_count after forced gender mismatch
            new_diff = (mismatch_attrs_np != original_attrs_np)[0]
            diff_count = new_diff.sum()

            # If we still need more mismatches, proceed to flip attributes
            if diff_count < MIN_MISMATCH_ATTRS:
                same_indices = [idx for idx, val in enumerate(new_diff) if val == 0]

                needed_flips = MIN_MISMATCH_ATTRS - diff_count
                zero_zero_indices = [idx for idx in same_indices if mismatch_attrs_np[0, idx] == 0]

                if len(zero_zero_indices) >= needed_flips:
                    flip_indices = np.random.choice(zero_zero_indices, size=needed_flips, replace=False)
                    mismatch_attrs_np[0, flip_indices] = 1
                else:
                    # Flip all zero_zero_indices
                    mismatch_attrs_np[0, zero_zero_indices] = 1
                    flips_done = len(zero_zero_indices)
                    still_needed = needed_flips - flips_done
                    remaining_same = [idx for idx in same_indices if idx not in zero_zero_indices]
                    if len(remaining_same) >= still_needed:
                        flip_indices = np.random.choice(remaining_same, size=still_needed, replace=False)
                        # Flip these by inverting them
                        mismatch_attrs_np[0, flip_indices] = 1 - mismatch_attrs_np[0, flip_indices]
                    else:
                        # Flip whatever remains
                        for idx2 in remaining_same:
                            if still_needed == 0:
                                break
                            mismatch_attrs_np[0, idx2] = 1 - mismatch_attrs_np[0, idx2]
                            still_needed -= 1

            # Ensure at least one mismatch from each group
            # Check difference again
            new_diff = (mismatch_attrs_np != original_attrs_np)[0]
            for g in groups:
                if not any(new_diff[idx] for idx in g):
                    # Flip one attribute from this group
                    group_same = [idx for idx in g if mismatch_attrs_np[0, idx] == original_attrs_np[0, idx]]
                    if group_same:
                        chosen = np.random.choice(group_same)
                        mismatch_attrs_np[0, chosen] = 1 - mismatch_attrs_np[0, chosen]

            # If originally male, ensure female now, if originally female, ensure male now
            if original_male == 1 and original_female == 0:
                mismatch_attrs_np[0, 1] = 0  # male
                mismatch_attrs_np[0, 2] = 1  # female
            elif original_female == 1 and original_male == 0:
                mismatch_attrs_np[0, 2] = 0  # female
                mismatch_attrs_np[0, 1] = 1  # male

            mismatched_attrs = torch.tensor(mismatch_attrs_np, dtype=torch.float32).to(device)

            # Encode and compute KL divergences with final mismatched_attrs
            z_img, img_mu, img_logvar = model.encode_image(image)
            z_text, text_mu, text_logvar = model.encode_text(attributes)
            z_mismatched, mis_mu, mis_logvar = model.encode_text(mismatched_attrs)

            kl_match = torch.mean(0.5 * (
                (text_logvar - img_logvar) +
                (torch.exp(img_logvar) + (img_mu - text_mu)**2) / torch.exp(text_logvar) - 1
            )).item()

            kl_mismatch = torch.mean(0.5 * (
                (mis_logvar - img_logvar) +
                (torch.exp(img_logvar) + (img_mu - mis_mu)**2) / torch.exp(mis_logvar) - 1
            )).item()

            match_score = 1 / (1 + kl_match)
            mismatch_score = 1 / (1 + kl_mismatch)

            matching_scores.append(match_score)
            mismatched_scores.append(mismatch_score)

            if match_score > threshold and match_score > mismatch_score:
                correct_classifications += 1
            total_pairs += 1

    return {
        'matching_mean': float(np.mean(matching_scores)),
        'matching_std': float(np.std(matching_scores)),
        'mismatched_mean': float(np.mean(mismatched_scores)),
        'mismatched_std': float(np.std(mismatched_scores)),
        'accuracy': correct_classifications / total_pairs,
        'threshold': threshold,
        'correct': correct_classifications,
        'total': total_pairs
    }

def calculate_consistency_metrics(model, dataset, device, num_samples=1000):
    num_samples = min(num_samples, len(dataset))
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    
    model.eval()
    matching_scores = []
    mismatched_scores = []
    MIN_MISMATCH_ATTRS = 3
    
    with torch.no_grad():
        mismatched_indices = np.roll(indices, 1)
        
        for idx, mismatch_idx in zip(indices, mismatched_indices):
            sample = dataset[idx]
            mismatched_sample = dataset[mismatch_idx]
            
            # Process matching pair
            image = sample['image'].unsqueeze(0).to(device)
            attributes = sample['attributes'].unsqueeze(0).to(device)
            
            # Prepare mismatched attributes
            mismatched_attrs = mismatched_sample['attributes'].unsqueeze(0).to(device)
            diff_count = (mismatched_attrs != attributes).sum().item()
            if diff_count < MIN_MISMATCH_ATTRS:
                mismatch_attrs_np = mismatched_attrs.cpu().numpy().copy()
                original_attrs_np = attributes.cpu().numpy().copy()
                same_indices = np.where(mismatch_attrs_np == original_attrs_np)[1]
                needed_flips = MIN_MISMATCH_ATTRS - diff_count
                if len(same_indices) >= needed_flips:
                    flip_indices = np.random.choice(same_indices, size=needed_flips, replace=False)
                    mismatch_attrs_np[0, flip_indices] = 1 - mismatch_attrs_np[0, flip_indices]
                    mismatched_attrs = torch.tensor(mismatch_attrs_np, dtype=torch.float32).to(device)
            
            # Get encodings
            z_img, img_mu, img_logvar = model.encode_image(image)
            z_text, text_mu, text_logvar = model.encode_text(attributes)
            z_mismatched, mis_mu, mis_logvar = model.encode_text(mismatched_attrs)
            
            # Calculate KL divergences
            kl_match = torch.mean(0.5 * (
                (text_logvar - img_logvar) +
                (torch.exp(img_logvar) + (img_mu - text_mu)**2) / torch.exp(text_logvar) -
                1
            ))
            
            kl_mismatch = torch.mean(0.5 * (
                (mis_logvar - img_logvar) +
                (torch.exp(img_logvar) + (img_mu - mis_mu)**2) / torch.exp(mis_logvar) -
                1
            ))
            
            # Convert to similarity scores
            match_score = (1 / (1 + kl_match)).cpu().item()
            mismatch_score = (1 / (1 + kl_mismatch)).cpu().item()

            matching_scores.append(match_score)
            mismatched_scores.append(mismatch_score)
    
    # Calculate statistics
    matching_mean = np.mean(matching_scores)
    matching_std = np.std(matching_scores)
    mismatched_mean = np.mean(mismatched_scores)
    mismatched_std = np.std(mismatched_scores)
    
    weight_matching = 0.6 
    optimal_threshold = (weight_matching * matching_mean + 
                        (1 - weight_matching) * mismatched_mean)
    
    correct = 0
    total = len(matching_scores) * 2
    
    for match_score, mismatch_score in zip(matching_scores, mismatched_scores):
        if match_score > optimal_threshold:
            correct += 1
        if mismatch_score <= optimal_threshold:
            correct += 1
            
    accuracy = correct / total
    
    return {
        'matching_mean': matching_mean,
        'matching_std': matching_std,
        'mismatched_mean': mismatched_mean,
        'mismatched_std': mismatched_std,
        'optimal_threshold': optimal_threshold,
        'accuracy': accuracy,
        'all_matching_scores': matching_scores,
        'all_mismatched_scores': mismatched_scores
    }

def visualize_consistency_pairs(model, dataset, device, num_samples=3, save_dir='eval_results'):
    import textwrap
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    
    # Calculate consistency metrics to get the optimal threshold
    metrics = calculate_consistency_metrics(model, dataset, device)
    optimal_threshold = metrics['optimal_threshold']
    
    print(f"\nVisualization using threshold: {optimal_threshold:.3f}")
    print(f"Scores > {optimal_threshold:.3f} will be shown as matches (green)")
    print(f"Scores ≤ {optimal_threshold:.3f} will be shown as mismatches (red)")

    np.random.seed(None)
    
    # Find indices of fully correct samples
    max_check = min(1000, len(dataset))
    correct_indices = []
    
    with torch.no_grad():
        for i in range(max_check):
            sample = dataset[i]
            image = sample['image'].unsqueeze(0).to(device)
            attributes = sample['attributes'].unsqueeze(0).to(device)

            mismatch_idx = (i + 1) % len(dataset)
            mismatched_sample = dataset[mismatch_idx]
            mismatched_attrs = mismatched_sample['attributes'].unsqueeze(0).to(device)

            # Encode original pair
            z_img, img_mu, img_logvar = model.encode_image(image)
            z_text, text_mu, text_logvar = model.encode_text(attributes)

            # Encode mismatched pair
            z_mismatched, mis_mu, mis_logvar = model.encode_text(mismatched_attrs)

            kl_match = torch.mean(0.5 * (
                (text_logvar - img_logvar) +
                (torch.exp(img_logvar) + (img_mu - text_mu)**2) / torch.exp(text_logvar) - 
                1
            ))

            kl_mismatch = torch.mean(0.5 * (
                (mis_logvar - img_logvar) +
                (torch.exp(img_logvar) + (img_mu - mis_mu)**2) / torch.exp(mis_logvar) - 
                1
            ))

            match_score = (1 / (1 + kl_match)).cpu().item()
            mismatch_score = (1 / (1 + kl_mismatch)).cpu().item()

            if match_score > optimal_threshold and mismatch_score <= optimal_threshold:
                correct_indices.append(i)
    
    if len(correct_indices) < num_samples:
        print("Not enough fully correct samples found. Using all we have.")
        num_samples = min(num_samples, len(correct_indices))
    
    if num_samples == 0:
        print("No fully correct samples found. Visualization aborted.")
        return metrics

    indices = np.random.choice(correct_indices, num_samples, replace=False)
    mismatched_indices = np.roll(indices, 1)
    
    fig, axes = plt.subplots(1, num_samples*2, figsize=(30, 28), constrained_layout=True)
    plt.subplots_adjust(wspace=0.1, bottom=0.2, top=0.9)

    MIN_MISMATCH_ATTRS = 4
    group1 = [1, 2]       # male, female
    group2 = [3, 4, 9]    # smiling, eyeglasses, wearing_lipstick
    group3 = [5, 6, 7]    # black_hair, blond_hair, bald
    group4 = [8]          # mustache
    groups = [group1, group2, group3, group4]

    with torch.no_grad():
        for i, (idx, mismatch_idx) in enumerate(zip(indices, mismatched_indices)):
            sample = dataset[idx]
            image = sample['image'].unsqueeze(0).to(device)
            attributes = sample['attributes'].unsqueeze(0).to(device)
            caption = sample['caption']
            
            # Encode original pair
            z_img, img_mu, img_logvar = model.encode_image(image)
            z_text, text_mu, text_logvar = model.encode_text(attributes)
            
            kl_match = torch.mean(0.5 * (
                (text_logvar - img_logvar) +
                (torch.exp(img_logvar) + (img_mu - text_mu)**2) / torch.exp(text_logvar) - 
                1
            ))
            match_score = 1 / (1 + kl_match)
            
            ax = axes[i*2]
            show_img = (image[0].cpu() + 1) / 2
            ax.imshow(show_img.permute(1,2,0).numpy())
            ax.axis('off')
            
            wrapped_caption = '\n'.join(textwrap.wrap(caption, width=16))
            ax.text(0.5, 1.08, wrapped_caption, 
                    ha='center', va='bottom',
                    transform=ax.transAxes,
                    fontsize=32,
                    bbox=dict(facecolor='white', edgecolor='black', pad=5, alpha=0.9),
                    linespacing=1.4)

            ax.text(0.5, -0.12, f"Score: {match_score:.3f}",
                    ha='center', va='center',
                    transform=ax.transAxes,
                    fontsize=28, fontweight='bold', color='green' if match_score > optimal_threshold else 'red')

            ax.text(0.5, -0.25,
                    'MATCH' if match_score > optimal_threshold else 'MISMATCH',
                    ha='center', va='center',
                    transform=ax.transAxes,
                    fontsize=28, fontweight='bold', color='green' if match_score > optimal_threshold else 'red')
            
            # Mismatched pair
            mismatched_sample = dataset[mismatch_idx]
            mismatched_attrs = mismatched_sample['attributes'].unsqueeze(0).to(device)

            # Apply mismatch logic
            original_attrs_np = attributes.cpu().numpy().copy()
            original_male = original_attrs_np[0, 1]
            original_female = original_attrs_np[0, 2]

            mismatch_attrs_np = mismatched_attrs.cpu().numpy().copy()
            
            # Force gender mismatch
            if original_male == 1 and original_female == 0:
                mismatch_attrs_np[0, 1] = 0  # male
                mismatch_attrs_np[0, 2] = 1  # female
            elif original_female == 1 and original_male == 0:
                mismatch_attrs_np[0, 2] = 0  # female
                mismatch_attrs_np[0, 1] = 1  # male
            else:
                mismatch_attrs_np[0, 1] = 1
                mismatch_attrs_np[0, 2] = 0

            new_diff = (mismatch_attrs_np != original_attrs_np)[0]
            diff_count = new_diff.sum()

            if diff_count < MIN_MISMATCH_ATTRS:
                same_indices = [idx_for_flip for idx_for_flip, val in enumerate(new_diff) if val == 0]
                needed_flips = MIN_MISMATCH_ATTRS - diff_count
                zero_zero_indices = [idx_for_flip for idx_for_flip in same_indices if mismatch_attrs_np[0, idx_for_flip] == 0]

                if len(zero_zero_indices) >= needed_flips:
                    flip_indices = np.random.choice(zero_zero_indices, size=needed_flips, replace=False)
                    mismatch_attrs_np[0, flip_indices] = 1
                else:
                    mismatch_attrs_np[0, zero_zero_indices] = 1
                    flips_done = len(zero_zero_indices)
                    still_needed = needed_flips - flips_done
                    remaining_same = [x for x in same_indices if x not in zero_zero_indices]
                    if len(remaining_same) >= still_needed:
                        flip_indices = np.random.choice(remaining_same, size=still_needed, replace=False)
                        mismatch_attrs_np[0, flip_indices] = 1 - mismatch_attrs_np[0, flip_indices]
                    else:
                        for idx2 in remaining_same:
                            if still_needed == 0:
                                break
                            mismatch_attrs_np[0, idx2] = 1 - mismatch_attrs_np[0, idx2]
                            still_needed -= 1

            # Ensure at least one mismatch from each group
            new_diff = (mismatch_attrs_np != original_attrs_np)[0]
            for g in groups:
                if not any(new_diff[x] for x in g):
                    group_same = [x for x in g if mismatch_attrs_np[0, x] == original_attrs_np[0, x]]
                    if group_same:
                        chosen = np.random.choice(group_same)
                        mismatch_attrs_np[0, chosen] = 1 - mismatch_attrs_np[0, chosen]

            # Re-check gender mismatch
            if original_male == 1 and original_female == 0:
                mismatch_attrs_np[0, 1] = 0
                mismatch_attrs_np[0, 2] = 1
            elif original_female == 1 and original_male == 0:
                mismatch_attrs_np[0, 2] = 0
                mismatch_attrs_np[0, 1] = 1

            mismatched_attrs = torch.tensor(mismatch_attrs_np, dtype=torch.float32).to(device)

            # Decode the mismatched caption
            mismatched_caption = model._attributes_to_text(mismatched_attrs)[0]

            z_mismatched, mis_mu, mis_logvar = model.encode_text(mismatched_attrs)
            kl_mismatch = torch.mean(0.5 * (
                (mis_logvar - img_logvar) +
                (torch.exp(img_logvar) + (img_mu - mis_mu)**2) / torch.exp(mis_logvar) - 
                1
            ))
            mismatch_score = 1 / (1 + kl_mismatch)
            
            ax = axes[i*2 + 1]
            ax.imshow(show_img.permute(1,2,0).numpy())
            ax.axis('off')
            
            wrapped_mismatch_caption = '\n'.join(textwrap.wrap(mismatched_caption, width=25))
            # Place mismatch caption above image
            ax.text(0.5, 1.08, wrapped_mismatch_caption,
                    ha='center', va='bottom',
                    transform=ax.transAxes,
                    fontsize=32,
                    bbox=dict(facecolor='white', edgecolor='black', pad=5, alpha=0.9),
                    linespacing=1.4)
            
            score_color = 'green' if mismatch_score > optimal_threshold else 'red'
            ax.text(0.5, -0.12, f'Score: {mismatch_score:.3f}',
                    ha='center', va='center',
                    transform=ax.transAxes,
                    fontsize=28, fontweight='bold', color=score_color)
            
            ax.text(0.5, -0.25,
                    'MATCH' if mismatch_score > optimal_threshold else 'MISMATCH',
                    ha='center', va='center',
                    transform=ax.transAxes,
                    fontsize=28, fontweight='bold', color=score_color)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'consistency_pairs.png'), bbox_inches='tight', dpi=300, pad_inches=0.5)
    plt.close(fig)
    
    return metrics

def visualize_consistency_confusion_matrix(metrics, save_dir='eval_results'):
    os.makedirs(save_dir, exist_ok=True)
    
    matching_scores = metrics['all_matching_scores']
    mismatched_scores = metrics['all_mismatched_scores']
    threshold = metrics['optimal_threshold']
    
    true_labels = ['Match'] * len(matching_scores) + ['Mismatch'] * len(mismatched_scores)
    predicted_labels = []
    
    for score in matching_scores:
        predicted_labels.append('Match' if score > threshold else 'Mismatch')
    
    for score in mismatched_scores:
        predicted_labels.append('Match' if score > threshold else 'Mismatch')
    
    labels = ['Match', 'Mismatch']
    cm = confusion_matrix(true_labels, predicted_labels, labels=labels)
    
    tn, fp, fn, tp = cm.ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.title('Consistency Classification Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'consistency_confusion_matrix.png'))
    plt.close()
    
    print("\nConfusion Matrix Analysis:")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1 Score: {f1:.3f}")
    
    print("\nDetailed Results:")
    print(f"True Positives (Correct Matches): {tp}")
    print(f"True Negatives (Correct Mismatches): {tn}")
    print(f"False Positives (Incorrect Matches): {fp}")
    print(f"False Negatives (Incorrect Mismatches): {fn}")
    
    return {
        'confusion_matrix': cm,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def visualize_paired_latent_spaces_2d(model, dataset, device, num_samples=200, save_dir='eval_results', perplexity=30):
    import os
    import numpy as np
    import torch
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    from matplotlib.patches import ConnectionPatch
    import seaborn as sns

    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    # Set global font size (adjust as needed)
    plt.rcParams['font.size'] = 14

    # Collect embeddings
    z_images = []
    z_texts = []
    labels = []

    # Find index for male attribute
    male_idx = None
    for k, v in model.idx_to_attribute.items():
        if v == 'male':
            male_idx = k
            break

    if male_idx is None:
        raise ValueError("Attribute 'male' not found in model.idx_to_attribute")

    # Collect samples
    with torch.no_grad():
        for i in range(min(num_samples, len(dataset))):
            sample = dataset[i]
            image = sample['image'].unsqueeze(0).to(device)
            attrs = sample['attributes'].unsqueeze(0).to(device)

            z_img, _, _ = model.encode_image(image)
            z_txt, _, _ = model.encode_text(attrs)

            z_images.append(z_img.cpu().numpy()[0])
            z_texts.append(z_txt.cpu().numpy()[0])

            label = "male" if attrs[0, male_idx].item() == 1 else "female"
            labels.append(label)

    z_images = np.array(z_images)
    z_texts = np.array(z_texts)

    # Compute t-SNE embeddings separately
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=1000, random_state=42)
    z_img_tsne = tsne.fit_transform(z_images)
    z_txt_tsne = tsne.fit_transform(z_texts)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={'wspace': 0.3})

    colors = {'male': '#2ecc71', 'female': '#e74c3c'}

    # Plot image embeddings
    for label in colors:
        mask = np.array(labels) == label
        ax1.scatter(
            z_img_tsne[mask, 0],
            z_img_tsne[mask, 1],
            c=colors[label],
            label=f'Image ({label})',
            alpha=0.7,
            s=80
        )
    ax1.set_title('Image Embeddings', fontsize=16)
    ax1.set_xlabel('t-SNE 1', fontsize=14)
    ax1.set_ylabel('t-SNE 2', fontsize=14)
    ax1.legend(fontsize=12)
    ax1.tick_params(axis='both', labelsize=12)
    ax1.grid(True, alpha=0.3)

    # Plot text embeddings
    for label in colors:
        mask = np.array(labels) == label
        ax2.scatter(
            z_txt_tsne[mask, 0],
            z_txt_tsne[mask, 1],
            c=colors[label],
            label=f'Text ({label})',
            alpha=0.7,
            s=80
        )
    ax2.set_title('Text Embeddings', fontsize=16)
    ax2.set_xlabel('t-SNE 1', fontsize=14)
    ax2.set_ylabel('t-SNE 2', fontsize=14)
    ax2.legend(fontsize=12)
    ax2.tick_params(axis='both', labelsize=12)
    ax2.grid(True, alpha=0.3)

    num_lines = min(30, num_samples)
    for i in range(num_lines):
        line_color = colors[labels[i]]
        con = ConnectionPatch(
            xyA=(z_img_tsne[i, 0], z_img_tsne[i, 1]),
            xyB=(z_txt_tsne[i, 0], z_txt_tsne[i, 1]),
            coordsA="data",
            coordsB="data",
            axesA=ax1,
            axesB=ax2,
            color=line_color,
            alpha=0.4,
            linestyle="--",
            linewidth=0.8
        )
        fig.add_artist(con)

    plt.suptitle('Cross-Modal Latent Space Alignment', fontsize=18, y=1.02)

    plt.savefig(
        os.path.join(save_dir, 'paired_tsne_2d_gender_colored_lines.png'),
        dpi=300,
        bbox_inches='tight',
        pad_inches=0.5
    )
    plt.close()

    return {
        'num_samples': num_samples,
        'num_lines_shown': num_lines,
        'perplexity': perplexity,
        'label_distribution': {label: labels.count(label) for label in set(labels)}
    }
    
def visualize_image_to_text_generation(model, dataset, device, num_samples=4, save_dir='eval_results'):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    
    total_samples = len(dataset)
    random_indices = np.random.choice(total_samples, num_samples, replace=False)
    
    for i, idx in enumerate(random_indices):
        sample = dataset[idx]
        image = sample['image'].unsqueeze(0).to(device)
        gt_caption = sample['caption']
        
        with torch.no_grad():
            generated_text = model.generate_from_image(image)
        
        fig, axes = plt.subplots(1, 2, figsize=(8, 3))
        
        # Ground truth image
        show_img = (image[0].cpu() + 1) / 2
        axes[0].imshow(show_img.permute(1, 2, 0).numpy())
        axes[0].set_title("Original Image", fontsize=14)
        axes[0].axis('off')
        
        # Generated text
        wrapped_text = '\n'.join(wrap(generated_text, width=40))
        axes[1].text(0.5, 0.5, wrapped_text, ha='center', va='center', 
                     fontsize=16, bbox=dict(facecolor='white', alpha=0.9))
        axes[1].set_title("Generated Text", fontsize=14)
        axes[1].axis('off')
        
        plt.tight_layout()
        filename = os.path.join(save_dir, f'image_to_text_sample_{i}.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)

def visualize_text_to_image_generation(model, dataset, device, num_samples=4, save_dir='eval_results'):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    
    total_samples = len(dataset)
    random_indices = np.random.choice(total_samples, num_samples, replace=False)
    
    for i, idx in enumerate(random_indices):
        sample = dataset[idx]
        attributes = sample['attributes'].unsqueeze(0).to(device)
        gt_caption = sample['caption']
        
        with torch.no_grad():
            generated_image = model.generate_from_text(attributes)[0].cpu()
        
        fig, axes = plt.subplots(1, 2, figsize=(8, 3))
        
        # Ground truth text
        wrapped_caption = '\n'.join(wrap(gt_caption, width=40))
        axes[0].text(0.5, 0.5, wrapped_caption, ha='center', va='center', 
                     fontsize=16, bbox=dict(facecolor='white', alpha=0.9))
        axes[0].set_title("Ground Truth Text", fontsize=14)
        axes[0].axis('off')
        
        # Generated image
        show_gen_img = (generated_image + 1) / 2
        axes[1].imshow(show_gen_img.permute(1, 2, 0).numpy())
        axes[1].set_title("Generated Image", fontsize=14)
        axes[1].axis('off')
        
        plt.tight_layout()
        filename = os.path.join(save_dir, f'text_to_image_sample_{i}.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)
    
def evaluate_train_val(model, device, train_subset, val_subset):
    train_scores = evaluate_consistency(model, train_subset, device)
    print("Train Set Consistency Accuracy:", train_scores['accuracy'])

    val_scores = evaluate_consistency(model, val_subset, device)
    print("Validation Set Consistency Accuracy:", val_scores['accuracy'])
    
def evaluate_main(latent_dim=256, model_checkpoint='checkpoints/final_model_256.pt'):
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    random.seed(config['seed'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device for evaluation: {device}")

    # Load model and dataset using the provided latent_dim and model_checkpoint
    model, full_dataset = load_model_and_data(device, latent_dim, model_checkpoint=model_checkpoint)

    dataset_size = len(full_dataset)
    val_size = int(config['val_split'] * dataset_size)
    train_size = dataset_size - val_size

    train_subset, val_subset = torch.utils.data.random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(config['seed'])
    )

    evaluate_train_val(model, device, train_subset, val_subset)
    
    # confusion matrix for consistency pairs
    metrics = visualize_consistency_pairs(model, val_subset, device, num_samples=3)
    visualize_consistency_confusion_matrix(metrics)
    scores = evaluate_consistency(model, val_subset, device)
    print(f"Consistency Classification Accuracy: {scores['accuracy']:.3f}")
    print(f"Threshold used: {scores['threshold']}")
    print(f"Correct classifications: {scores['correct']}/{scores['total']}")
    
    # Visualize pairs with scores
    #visualize_consistency_pairs(model, val_subset, device)
    
    #visualize_paired_latent_spaces_2d(model, val_subset, device, num_samples=500, save_dir='eval_results')
        
    # Visualize 5 samples of image-to-text generation
    #visualize_image_to_text_generation(model, val_subset, device, num_samples=10, save_dir='eval_results')

    # Visualize 5 samples of text-to-image generation
    #visualize_text_to_image_generation(model, val_subset, device, num_samples=10, save_dir='eval_results')
    
    #visualize_random_examples(full_dataset, num_samples=5)
    
if __name__ == "__main__":
    # consistency checking scores
    #evaluate_main(latent_dim=128, model_checkpoint='checkpoints/final_model_128.pt')
    evaluate_main(latent_dim=256, model_checkpoint='checkpoints/final_model_256.pt')
    #evaluate_main(latent_dim=512, model_checkpoint='checkpoints/final_model_512.pt')