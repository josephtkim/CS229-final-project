import torch
import os
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import torch.nn.functional as F

from textwrap import wrap
import textwrap


from config import config
from data import CelebADataset, get_celeba_subset
from models import MultimodalVAE
from utils import clean_and_validate_attributes
from visualization import VisualizationUtils

def load_model_and_data(device, model_checkpoint='final_model.pt'):
    checkpoint = torch.load(model_checkpoint, map_location=device)
    model = MultimodalVAE(
        latent_dim=config['latent_dim'],
        temperature=1.0
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()

    test_dataset = get_celeba_subset(
        root_dir=config['data_path'],
        subset_size=10000,
        random_subset=True,
        cache_path=config['cache_path']
    )
    dataset = CelebADataset(test_dataset)
    return model, dataset
    
def visualize_bidirectional_generation(model, dataset, device, num_samples=4, save_dir='eval_results'):
    """
    Visualizes text-to-image and image-to-text generation alongside ground truth data.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    print("Generating bidirectional samples...")
    
    for sample_idx in range(num_samples):
        sample = dataset[sample_idx]
        image = sample['image'].unsqueeze(0).to(device)
        attributes = sample['attributes'].unsqueeze(0).to(device)
        gt_caption = sample['caption']
        
        with torch.no_grad():
            generated_text = model.generate_from_image(image)
            generated_image = model.generate_from_text(attributes)[0].cpu()
            
        fig = plt.figure(figsize=(8, 5))
        gs = plt.GridSpec(2, 2, height_ratios=[3, 1], hspace=0.3, wspace=0.3)
        
        ax1 = plt.subplot(gs[0, 0])
        show_gt_img = (image[0].cpu() + 1) / 2
        ax1.imshow(show_gt_img.permute(1,2,0).numpy())
        ax1.set_title("Ground Truth", fontsize=10)
        ax1.axis('off')
        
        ax2 = plt.subplot(gs[0, 1])
        show_gen_img = (generated_image + 1) / 2
        ax2.imshow(show_gen_img.permute(1,2,0).numpy())
        ax2.set_title("Generated", fontsize=10)
        ax2.axis('off')
        
        ax3 = plt.subplot(gs[1, 0])
        ax3.text(0.5, 0.5, gt_caption, 
                ha='center', va='center', wrap=True,
                fontsize=8,
                bbox=dict(facecolor='white', alpha=0.8))
        ax3.axis('off')
        
        ax4 = plt.subplot(gs[1, 1])
        ax4.text(0.5, 0.5, generated_text, 
                ha='center', va='center', wrap=True,
                fontsize=8,
                bbox=dict(facecolor='white', alpha=0.8))
        ax4.axis('off')
        
        plt.savefig(os.path.join(save_dir, f'bidirectional_generation_sample_{sample_idx}.png'), 
                   dpi=300, bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)


def evaluate_attributes(model, dataset, device):
    model.eval()
    all_preds = []
    all_labels = []
    for i in range(len(dataset)):
        sample = dataset[i]
        image = sample['image'].unsqueeze(0).to(device)
        attributes = sample['attributes'].unsqueeze(0)
        with torch.no_grad():
            outputs = model(images=image, target_attributes=attributes.to(device))
            preds = (outputs['recon_text_probs'] > 0.5).float().cpu().numpy()
            labels = attributes.cpu().numpy()

        all_preds.append(preds[0])
        all_labels.append(labels[0])

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    all_preds_flat = all_preds.flatten()
    all_labels_flat = all_labels.flatten()

    cm = confusion_matrix(all_labels_flat, all_preds_flat)
    global_f1 = f1_score(all_labels_flat, all_preds_flat)

    print("Global Confusion Matrix across all attributes:")
    print(cm)
    print(f"Global F1 Score across all attributes: {global_f1:.4f}")

    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap='Blues')
    plt.title("Global Confusion Matrix (All Attributes Combined)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig("eval_results/global_confusion_matrix.png")
    plt.close()
    
def visualize_random_examples(dataset, num_samples=5, save_dir='eval_results'):
    os.makedirs(save_dir, exist_ok=True)
    
    total_samples = len(dataset)
    random_indices = np.random.choice(total_samples, num_samples, replace=False)
    
    fig, axes = plt.subplots(1, num_samples, figsize=(2.5*num_samples, 3))
    
    if num_samples == 1:
        axes = [axes]
    
    for idx, ax in enumerate(axes):
        sample = dataset[random_indices[idx]]
        image = sample['image']
        caption = sample['caption']
        
        show_img = (image + 1) / 2
        ax.imshow(show_img.permute(1,2,0).numpy())
        ax.axis('off')
        
        wrapped_caption = '\n'.join(wrap(caption, width=30)) 
        ax.text(0.5, -0.05, wrapped_caption,
               ha='center', va='top',
               transform=ax.transAxes,
               wrap=True,
               fontsize=8)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2) 
    plt.savefig(os.path.join(save_dir, 'random_dataset_examples.png'),
                bbox_inches='tight',
                dpi=300)
    plt.close(fig)


def visualize_latent_space(model, dataset, device, num_visualizations=4, save_dir='eval_results'):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    num_samples = min(len(dataset), 200)
    z_images = []
    labels = []

    male_idx = None
    for k, v in model.idx_to_attribute.items():
        if v == 'male':
            male_idx = k
            break

    if male_idx is None:
        raise ValueError("Attribute 'male' not found in model.idx_to_attribute.")

    # Collect embeddings
    for i in range(num_samples):
        sample = dataset[i]
        image = sample['image'].unsqueeze(0).to(device)
        attrs = sample['attributes'].unsqueeze(0).to(device)
        with torch.no_grad():
            z_img, _, _ = model.encode_image(image)
        z_images.append(z_img.cpu().numpy()[0])
        label = "male" if attrs[0, male_idx].item() == 1 else "female"
        labels.append(label)

    z_images = np.array(z_images)

    # Generate visualization
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000)
    z_img_tsne = tsne.fit_transform(z_images)

    # Create visualization
    plt.figure(figsize=(5, 4))
    for l in set(labels):
        idxs = [i for i, x in enumerate(labels) if x == l]
        plt.scatter(z_img_tsne[idxs, 0], z_img_tsne[idxs, 1], 
                   label=l, alpha=0.7, s=30)
    plt.title("t-SNE Visualization of Image Latent Space", fontsize=10)
    plt.xlabel("t-SNE Dimension 1", fontsize=9)
    plt.ylabel("t-SNE Dimension 2", fontsize=9)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'tsne_latent_space.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    
def evaluate_consistency(model, dataset, device, num_samples=5000):
    """Evaluates consistency score for both matching and mismatched image-text pairs"""
    model.eval()
    matching_scores = []
    mismatched_scores = []
    
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    mismatched_indices = np.random.choice(len(dataset), num_samples, replace=False)
    
    for i, idx in enumerate(indices):
        sample = dataset[idx]
        mismatched_sample = dataset[mismatched_indices[i]]
        
        image = sample['image'].unsqueeze(0).to(device)
        matched_attributes = sample['attributes'].unsqueeze(0).to(device)
        mismatched_attributes = mismatched_sample['attributes'].unsqueeze(0).to(device)
        
        with torch.no_grad():
            z_image, _, _ = model.encode_image(image)
            z_matched_text, _, _ = model.encode_text(matched_attributes)
            z_mismatched_text, _, _ = model.encode_text(mismatched_attributes)
            
            z_image_norm = F.normalize(model.norm_layer(z_image), dim=-1)
            z_matched_norm = F.normalize(model.norm_layer(z_matched_text), dim=-1)
            z_mismatched_norm = F.normalize(model.norm_layer(z_mismatched_text), dim=-1)
            
            matching_score = F.cosine_similarity(z_image_norm, z_matched_norm)
            mismatched_score = F.cosine_similarity(z_image_norm, z_mismatched_norm)
            
            matching_scores.append(matching_score.item())
            mismatched_scores.append(mismatched_score.item())
    
    return {
        'matching': (np.mean(matching_scores), np.std(matching_scores)),
        'mismatched': (np.mean(mismatched_scores), np.std(mismatched_scores))
    }

    
def visualize_results_grid(model, dataset, device, num_examples=3, save_dir='eval_results', iteration=None):
    os.makedirs(save_dir, exist_ok=True)
    
    total_samples = len(dataset)
    random_indices = np.random.choice(total_samples, num_examples, replace=False)
    
    fig = plt.figure(figsize=(10, 5*num_examples))
    gs = plt.GridSpec(num_examples, 2, figure=fig, 
                     height_ratios=[1]*num_examples,
                     hspace=0.4, wspace=0.2)
    
    for idx, row in enumerate(range(num_examples)):
        sample = dataset[random_indices[idx]]
        image = sample['image'].unsqueeze(0).to(device)
        attributes = sample['attributes'].unsqueeze(0).to(device)
        gt_caption = sample['caption']
        
        with torch.no_grad():
            generated_text = model.generate_from_image(image)
            generated_image = model.generate_from_text(attributes)[0].cpu()
        
        # Ground Truth Column
        ax_gt = fig.add_subplot(gs[row, 0])
        show_gt_img = (image[0].cpu() + 1) / 2
        ax_gt.imshow(show_gt_img.permute(1,2,0).numpy())
        ax_gt.axis('off')
        
        wrapped_gt_text = '\n'.join(textwrap.wrap(gt_caption, width=40))
        ax_gt.text(0.5, -0.15, wrapped_gt_text,
                  ha='center', va='top',
                  wrap=True, fontsize=12,
                  transform=ax_gt.transAxes,
                  bbox=dict(facecolor='white', alpha=0.9, pad=3,
                          edgecolor='none'),
                  linespacing=1.2)
        if row == 0:
            ax_gt.set_title('Ground Truth', fontsize=12, pad=10)
            
        # Cross-Modal Generation Column
        ax_gen = fig.add_subplot(gs[row, 1])
        show_gen_img = (generated_image + 1) / 2
        ax_gen.imshow(show_gen_img.permute(1,2,0).numpy())
        ax_gen.axis('off')
        
        wrapped_gen_text = '\n'.join(textwrap.wrap(generated_text, width=40))
        ax_gen.text(0.5, -0.15, wrapped_gen_text,
                   ha='center', va='top',
                   wrap=True, fontsize=12,
                   transform=ax_gen.transAxes,
                   bbox=dict(facecolor='white', alpha=0.9, pad=3,
                           edgecolor='none'),
                   linespacing=1.2)
        if row == 0:
            ax_gen.set_title('Cross-Modal Generation', fontsize=12, pad=10)
    
    # Create filename based on iteration number
    if iteration is not None:
        filename = f'cross_modal_grid_{iteration:03d}.png'
    else:
        filename = 'cross_modal_grid.png'
        
    plt.savefig(os.path.join(save_dir, filename),
                bbox_inches='tight', dpi=300, pad_inches=0.3)
    plt.close(fig)

def visualize_unimodal_reconstructions(model, dataset, device, num_examples=3, save_dir='eval_results'):
    os.makedirs(save_dir, exist_ok=True)
    
    total_samples = len(dataset)
    random_indices = np.random.choice(total_samples, num_examples, replace=False)
    
    fig = plt.figure(figsize=(10, 5*num_examples))
    gs = plt.GridSpec(num_examples, 2, figure=fig, 
                     height_ratios=[1]*num_examples,
                     hspace=0.4, wspace=0.2)
    
    for idx, row in enumerate(range(num_examples)):
        sample = dataset[random_indices[idx]]
        image = sample['image'].unsqueeze(0).to(device)
        attributes = sample['attributes'].unsqueeze(0).to(device)
        gt_caption = sample['caption']
        
        with torch.no_grad():
            outputs = model(images=image, target_attributes=attributes)
            reconstructed_image = outputs['recon_images'][0].cpu()
            reconstructed_text = outputs['text_from_image_probs'][0]
            reconstructed_attributes = (reconstructed_text > 0.5).float()
            reconstructed_caption = model._attributes_to_text(reconstructed_attributes.unsqueeze(0))[0]
        
        # Ground Truth Column
        ax_gt = fig.add_subplot(gs[row, 0])
        show_gt_img = (image[0].cpu() + 1) / 2
        ax_gt.imshow(show_gt_img.permute(1,2,0).numpy())
        ax_gt.axis('off')
        
        wrapped_gt_text = '\n'.join(textwrap.wrap(gt_caption, width=40))
        ax_gt.text(0.5, -0.15, wrapped_gt_text,
                  ha='center', va='top',
                  wrap=True, fontsize=12,
                  transform=ax_gt.transAxes,
                  bbox=dict(facecolor='white', alpha=0.9, pad=3,
                          edgecolor='none'),
                  linespacing=1.2)
        if row == 0:
            ax_gt.set_title('Ground Truth', fontsize=12, pad=10)
            
        # Reconstruction Column
        ax_recon = fig.add_subplot(gs[row, 1])
        show_recon_img = (reconstructed_image + 1) / 2
        ax_recon.imshow(show_recon_img.permute(1,2,0).numpy())
        ax_recon.axis('off')
        
        # Wrap reconstructed text and add it with more space
        wrapped_recon_text = '\n'.join(textwrap.wrap(reconstructed_caption, width=40))
        ax_recon.text(0.5, -0.15, wrapped_recon_text,
                     ha='center', va='top',
                     wrap=True, fontsize=12,
                     transform=ax_recon.transAxes,
                     bbox=dict(facecolor='white', alpha=0.9, pad=3,
                             edgecolor='none'),
                     linespacing=1.2)
        if row == 0:
            ax_recon.set_title('Reconstructions', fontsize=12, pad=10)
    
    plt.savefig(os.path.join(save_dir, 'unimodal_reconstructions_grid.png'),
                bbox_inches='tight', dpi=300, pad_inches=0.3)
    plt.close(fig)
    
def evaluate_consistency(model, dataset, device, num_samples=100, threshold=0.20):
    model.eval()
    matching_scores = []
    mismatched_scores = []
    correct_classifications = 0
    total_pairs = 0
    
    with torch.no_grad():
        for i in range(num_samples):
            sample = dataset[i]
            image = sample['image'].unsqueeze(0).to(device)
            attributes = sample['attributes'].unsqueeze(0).to(device)
            
            # Get random mismatched pair
            mismatch_idx = (i + 1) % len(dataset)
            mismatched_sample = dataset[mismatch_idx]
            mismatched_attrs = mismatched_sample['attributes'].unsqueeze(0).to(device)
            
            # Get encodings and calculate KL divergence
            z_img, img_mu, img_logvar = model.encode_image(image)
            z_text, text_mu, text_logvar = model.encode_text(attributes)
            z_mismatched, mis_mu, mis_logvar = model.encode_text(mismatched_attrs)
            
            # Calculate KL divergences
            kl_match = torch.mean(0.5 * (
                torch.exp(text_logvar) / torch.exp(img_logvar) +
                (img_mu - text_mu)**2 / torch.exp(img_logvar) -
                1 + img_logvar - text_logvar
            )).item()
            
            kl_mismatch = torch.mean(0.5 * (
                torch.exp(mis_logvar) / torch.exp(img_logvar) +
                (img_mu - mis_mu)**2 / torch.exp(img_logvar) -
                1 + img_logvar - mis_logvar
            )).item()
            
            # Convert to similarity scores
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
    """
    Calculates consistency metrics across a subset of data to determine optimal threshold.
    """
    model.eval()
    matching_scores = []
    mismatched_scores = []
    
    with torch.no_grad():
        indices = np.random.choice(len(dataset), num_samples, replace=False)
        mismatched_indices = np.roll(indices, 1)
        
        for idx, mismatch_idx in zip(indices, mismatched_indices):
            sample = dataset[idx]
            mismatched_sample = dataset[mismatch_idx]
            
            # Process matching pair
            image = sample['image'].unsqueeze(0).to(device)
            attributes = sample['attributes'].unsqueeze(0).to(device)
            
            # Get encodings
            z_img, img_mu, img_logvar = model.encode_image(image)
            z_text, text_mu, text_logvar = model.encode_text(attributes)
            z_mismatched, mis_mu, mis_logvar = model.encode_text(
                mismatched_sample['attributes'].unsqueeze(0).to(device)
            )
            
            # Calculate KL divergences
            kl_match = torch.mean(0.5 * (
                torch.exp(text_logvar) / torch.exp(img_logvar) +
                (img_mu - text_mu)**2 / torch.exp(img_logvar) -
                1 + img_logvar - text_logvar
            )).item()
            
            kl_mismatch = torch.mean(0.5 * (
                torch.exp(mis_logvar) / torch.exp(img_logvar) +
                (img_mu - mis_mu)**2 / torch.exp(img_logvar) -
                1 + img_logvar - mis_logvar
            )).item()
            
            # Convert to similarity scores
            match_score = 1 / (1 + kl_match)
            mismatch_score = 1 / (1 + kl_mismatch)
            
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

def visualize_consistency_pairs(model, dataset, device, num_samples=4, save_dir='eval_results'):
    """
    Visualizes pairs using KL divergence-based consistency scores in a single row format.
    Each example shows the image, caption, similarity score, and match/mismatch label.
    """
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    
    metrics = calculate_consistency_metrics(model, dataset, device)
    optimal_threshold = metrics['optimal_threshold']
    
    print(f"\nVisualization using threshold: {optimal_threshold:.3f}")
    print(f"Scores > {optimal_threshold:.3f} will be shown as matches (green)")
    print(f"Scores ≤ {optimal_threshold:.3f} will be shown as mismatches (red)")
    
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    mismatched_indices = np.roll(indices, 1)
    
    fig, axes = plt.subplots(1, num_samples*2, figsize=(20, 5))
    plt.subplots_adjust(wspace=0.3)
    
    with torch.no_grad():
        for i, (idx, mismatch_idx) in enumerate(zip(indices, mismatched_indices)):
            sample = dataset[idx]
            image = sample['image'].unsqueeze(0).to(device)
            attributes = sample['attributes'].unsqueeze(0).to(device)
            caption = sample['caption']
            
            z_img, img_mu, img_logvar = model.encode_image(image)
            z_text, text_mu, text_logvar = model.encode_text(attributes)
            
            kl_match = torch.mean(0.5 * (
                torch.exp(text_logvar) / torch.exp(img_logvar) +
                (img_mu - text_mu)**2 / torch.exp(img_logvar) -
                1 + img_logvar - text_logvar
            )).item()
            match_score = 1 / (1 + kl_match)
            
            ax = axes[i*2]
            show_img = (image[0].cpu() + 1) / 2
            ax.imshow(show_img.permute(1,2,0).numpy())
            ax.axis('off')
            
            wrapped_caption = '\n'.join(textwrap.wrap(caption, width=30))
            ax.text(0.5, -0.1, wrapped_caption, 
                   ha='center', va='top', 
                   transform=ax.transAxes,
                   fontsize=8,
                   color='black')
                   
            score_color = 'green' if match_score > optimal_threshold else 'red'
            ax.text(0.5, -0.25, f'Score: {match_score:.3f}',
                   ha='center', va='top',
                   transform=ax.transAxes,
                   fontsize=8,
                   color=score_color)
            ax.text(0.5, -0.35, 'MATCH' if match_score > optimal_threshold else 'MISMATCH',
                   ha='center', va='top',
                   transform=ax.transAxes,
                   fontsize=8,
                   color=score_color)
            
            mismatched_sample = dataset[mismatch_idx]
            mismatched_attrs = mismatched_sample['attributes'].unsqueeze(0).to(device)
            mismatched_caption = mismatched_sample['caption']
            
            z_mismatched, mis_mu, mis_logvar = model.encode_text(mismatched_attrs)
            kl_mismatch = torch.mean(0.5 * (
                torch.exp(mis_logvar) / torch.exp(img_logvar) +
                (img_mu - mis_mu)**2 / torch.exp(img_logvar) -
                1 + img_logvar - mis_logvar
            )).item()
            mismatch_score = 1 / (1 + kl_mismatch)
            
            ax = axes[i*2 + 1]
            ax.imshow(show_img.permute(1,2,0).numpy())
            ax.axis('off')
            
            wrapped_caption = '\n'.join(textwrap.wrap(mismatched_caption, width=30))
            ax.text(0.5, -0.1, wrapped_caption,
                   ha='center', va='top',
                   transform=ax.transAxes,
                   fontsize=8,
                   color='black')
                   
            score_color = 'green' if mismatch_score > optimal_threshold else 'red'
            ax.text(0.5, -0.25, f'Score: {mismatch_score:.3f}',
                   ha='center', va='top',
                   transform=ax.transAxes,
                   fontsize=8,
                   color=score_color)
            ax.text(0.5, -0.35, 'MATCH' if mismatch_score > optimal_threshold else 'MISMATCH',
                   ha='center', va='top',
                   transform=ax.transAxes,
                   fontsize=8,
                   color=score_color)
    
    plt.savefig(os.path.join(save_dir, 'consistency_pairs.png'),
                bbox_inches='tight', dpi=300, pad_inches=0.5)
    plt.close(fig)
    
    return metrics
    
def visualize_consistency_confusion_matrix(metrics, save_dir='eval_results'):
    """
    Creates and visualizes a confusion matrix for the consistency scores.
    
    Args:
        metrics: Dictionary containing matching and mismatched scores
        save_dir: Directory to save the visualization
    """
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

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, dataset = load_model_and_data(device, model_checkpoint='final_model.pt')
    
    metrics = visualize_consistency_pairs(model, dataset, device)

    
    scores = evaluate_consistency(model, dataset, device)
    print(f"Consistency Classification Accuracy: {scores['accuracy']:.3f}")
    print(f"Threshold used: {scores['threshold']}")
    print(f"Correct classifications: {scores['correct']}/{scores['total']}")
    
    # Visualize pairs with scores
    visualize_consistency_pairs(model, dataset, device)
   
    #for i in range(1, 11):
    #    visualize_results_grid(model, dataset, device, iteration=i)
        
    #visualize_unimodal_reconstructions(model, dataset, device, num_examples=4)
    #evaluate_attributes(model, dataset, device)
    #visualize_generation_steps(model, dataset, device, num_samples=4)
    # visualize_latent_space(model, dataset, device, num_visualizations=4)
    #visualize_bidirectional_generation(model, dataset, device, num_samples=4)
    #visualize_random_examples(dataset, num_samples=5)