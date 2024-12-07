import torch
import os
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from textwrap import wrap
import textwrap

from config import config
from data import CelebADataset, get_celeba_subset
from models import MultimodalVAE
from utils import clean_and_validate_attributes
from visualization import VisualizationUtils

def load_model_and_data(device, model_checkpoint='final_model.pt'):
    # Load the trained model
    checkpoint = torch.load(model_checkpoint, map_location=device)
    model = MultimodalVAE(
        latent_dim=config['latent_dim'],
        temperature=1.0
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()

    # Load a subset of data for evaluation
    test_dataset = get_celeba_subset(
        root_dir=config['data_path'],
        subset_size=500,
        random_subset=True,
        cache_path=config['cache_path']
    )
    dataset = CelebADataset(test_dataset)
    return model, dataset
    
def visualize_bidirectional_generation(model, dataset, device, num_samples=4, save_dir='eval_results'):
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

    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000)
    z_img_tsne = tsne.fit_transform(z_images)

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
            
        ax_recon = fig.add_subplot(gs[row, 1])
        show_recon_img = (reconstructed_image + 1) / 2
        ax_recon.imshow(show_recon_img.permute(1,2,0).numpy())
        ax_recon.axis('off')
        
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

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, dataset = load_model_and_data(device, model_checkpoint='final_model.pt')
        
    for i in range(1, 11):
        visualize_results_grid(model, dataset, device, iteration=i)
        
    #visualize_unimodal_reconstructions(model, dataset, device, num_examples=4)
    #evaluate_attributes(model, dataset, device)
    #visualize_generation_steps(model, dataset, device, num_samples=4)
    # visualize_latent_space(model, dataset, device, num_visualizations=4)
    #visualize_bidirectional_generation(model, dataset, device, num_samples=4)
    #visualize_random_examples(dataset, num_samples=5)