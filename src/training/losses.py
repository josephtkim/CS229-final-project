import torch
import torch.nn.functional as F
from ..models.discriminator import PatchDiscriminator

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