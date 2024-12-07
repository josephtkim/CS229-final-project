import re
import torch
import torch.nn.functional as F
from torchvision.models import vgg16, VGG16_Weights
import textwrap

def remove_punc_special(text):
    return ''.join(char.lower() for char in text if char.isalpha() or char.isspace())

def clean_and_validate_attributes(text_list):
    valid_attributes = {
        'young', 'male', 'female', 'smiling', 'eyeglasses',
        'black hair', 'blond hair', 'bald', 'mustache', 'wearing lipstick'
    }
    cleaned_words = []
    for text in text_list:
        cleaned_text = remove_punc_special(text)
        cleaned_words.extend(cleaned_text.split())

    found_attributes = []
    i = 0
    while i < len(cleaned_words):
        if i < len(cleaned_words)-1:
            two_words = f"{cleaned_words[i]} {cleaned_words[i+1]}"
            if two_words in valid_attributes:
                found_attributes.append(two_words)
                i+=2
                continue
        if cleaned_words[i] in valid_attributes:
            found_attributes.append(cleaned_words[i])
        i+=1
    return found_attributes

def generate_natural_description(text):
    attributes = clean_and_validate_attributes(text) if not isinstance(text, list) else clean_and_validate_attributes(text)
    if not attributes:
        return "A person."

    unique_attributes = set(attributes)
    parts = ['a']

    if 'young' in unique_attributes:
        parts.append('young')
        unique_attributes.remove('young')

    if 'male' in unique_attributes and 'female' in unique_attributes:
        parts.append('male')
        unique_attributes.discard('male')
        unique_attributes.discard('female')
    elif 'male' in unique_attributes:
        parts.append('male')
        unique_attributes.discard('male')
    elif 'female' in unique_attributes:
        parts.append('female')
        unique_attributes.discard('female')
    else:
        parts.append('person')

    special_attrs = ['smiling', 'bald', 'wearing lipstick', 'wearing hat']
    special_parts = []
    for attr in special_attrs:
        if attr in unique_attributes:
            special_parts.append(attr)
            unique_attributes.remove(attr)

    if special_parts:
        parts.append('who is ' + ' and '.join(special_parts))

    if unique_attributes:
        parts.append('with')
        parts.append(' and '.join(unique_attributes))

    return ' '.join(parts) + '.'

def token_f1_loss(ground_truth, reconstructed):
    gt_tokens = set(clean_and_validate_attributes(ground_truth.split()))
    recon_tokens = set(clean_and_validate_attributes(reconstructed.split()))

    key_attrs = {'male', 'female', 'young'}
    key_weight = 2.0

    weighted_tp = sum(key_weight if t in key_attrs else 1.0 for t in gt_tokens & recon_tokens)
    weighted_fp = sum(key_weight if t in key_attrs else 1.0 for t in recon_tokens - gt_tokens)
    weighted_fn = sum(key_weight if t in key_attrs else 1.0 for t in gt_tokens - recon_tokens)

    precision = weighted_tp / (weighted_tp + weighted_fp) if (weighted_tp + weighted_fp)>0 else 0
    recall = weighted_tp / (weighted_tp + weighted_fn) if (weighted_tp + weighted_fn)>0 else 0
    f1_score = (2*precision*recall)/(precision+recall) if (precision+recall)>0 else 0
    return 1 - f1_score**2

def attribute_loss(ground_truth, reconstructed):
    gt_attributes = clean_and_validate_attributes(ground_truth.split())
    recon_attributes = clean_and_validate_attributes(reconstructed.split())
    intersection = len(set(gt_attributes) & set(recon_attributes))
    union = len(set(gt_attributes) | set(recon_attributes))
    jaccard_similarity = intersection / union if union>0 else 0.0
    return 1 - jaccard_similarity

def compute_distribution_matching_loss(mu1, logvar1, mu2, logvar2):
    var1 = torch.exp(logvar1)
    var2 = torch.exp(logvar2)
    kl_div = 0.5*torch.mean(logvar2 - logvar1 + (var1+(mu1 - mu2).pow(2))/var2 -1)
    return kl_div

def identity_consistency_loss(real_features, generated_features):
    real_features = F.normalize(real_features, p=2, dim=1)
    generated_features = F.normalize(generated_features, p=2, dim=1)
    return 1.0 - F.cosine_similarity(real_features, generated_features).mean()

def compute_gradient_penalty(discriminator, real_samples, fake_samples):
    alpha = torch.rand(real_samples.size(0),1,1,1,device=real_samples.device)
    interpolated = (alpha*real_samples+(1-alpha)*fake_samples).requires_grad_(True)
    d_interpolated = discriminator(interpolated)
    grad_outputs = torch.ones_like(d_interpolated, device=real_samples.device)

    gradients = torch.autograd.grad(
        outputs=d_interpolated,
        inputs=interpolated,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    gradients = gradients.view(gradients.size(0),-1)
    gradient_penalty = ((gradients.norm(2,dim=1)-1)**2).mean()
    return gradient_penalty

def discriminator_loss(discriminator, real_images, fake_images, epoch, phase_config):
    def add_instance_noise(images, phase_config):
        base_noise = phase_config.get('instance_noise',0.1)
        decay_factor = phase_config.get('noise_decay',0.9)
        noise_std = base_noise*(decay_factor**epoch)
        return images+torch.randn_like(images)*noise_std

    real_images = real_images.detach()
    fake_images = fake_images.detach()

    real_preds = discriminator(add_instance_noise(real_images, phase_config))
    fake_preds = discriminator(add_instance_noise(fake_images, phase_config))

    real_preds = torch.clamp(real_preds,0.0,1.0)
    fake_preds = torch.clamp(fake_preds,0.0,1.0)

    real_loss = F.binary_cross_entropy(real_preds, torch.ones_like(real_preds))
    fake_loss = F.binary_cross_entropy(fake_preds, torch.zeros_like(fake_preds))

    gradient_penalty = 0.0
    if phase_config.get('use_gradient_penalty',False):
        gradient_penalty = compute_gradient_penalty(discriminator, real_images, fake_images)
        gradient_penalty *= phase_config.get('gradient_penalty_weight',10.0)

    return (real_loss+fake_loss)*0.5 + gradient_penalty

def generator_loss(discriminator, fake_images):
    fake_preds = discriminator(fake_images)
    fake_preds = torch.clamp(fake_preds,0.0,1.0)
    return F.binary_cross_entropy(fake_preds, torch.ones_like(fake_preds))

def generator_loss_eval(discriminator, fake_samples):
    with torch.no_grad():
        fake_preds = discriminator(fake_samples)
        fake_preds = torch.clamp(fake_preds,0.0,1.0)
    return F.binary_cross_entropy(fake_preds, torch.ones_like(fake_preds))

def discriminator_loss_eval(discriminator, real_samples, fake_samples, phase_config):
    with torch.no_grad():
        real_preds = discriminator(real_samples)
        fake_preds = discriminator(fake_samples)

    real_preds = torch.clamp(real_preds,0.0,1.0)
    fake_preds = torch.clamp(fake_preds,0.0,1.0)

    real_loss = F.binary_cross_entropy(real_preds, torch.ones_like(real_preds))
    fake_loss = F.binary_cross_entropy(fake_preds, torch.zeros_like(fake_preds))

    gradient_penalty = 0.0
    if phase_config.get("use_gradient_penalty",False):
        with torch.enable_grad():
            real_samples.requires_grad=True
            fake_samples.requires_grad=True
            gradient_penalty = compute_gradient_penalty(discriminator, real_samples, fake_samples)
            gradient_penalty *= phase_config.get("gradient_penalty_weight",10.0)

    return (real_loss+fake_loss)*0.5+gradient_penalty
