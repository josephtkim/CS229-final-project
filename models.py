import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16, VGG16_Weights
from utils import identity_consistency_loss, generate_natural_description
from utils import compute_distribution_matching_loss
from utils import clean_and_validate_attributes
from utils import remove_punc_special

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.BatchNorm2d(channels)
        )

    def forward(self, x):
        return x + self.block(x)

class ResidualLinear(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LeakyReLU(0.2),
            nn.Linear(dim, dim)
        )

    def forward(self, x):
        return x + self.block(x)

class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels=3, ndf=64):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, ndf, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(ndf, ndf * 2, 4, stride=2, padding=1),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2),

            nn.Conv2d(ndf * 2, ndf * 4, 4, stride=2, padding=1),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2),

            nn.Conv2d(ndf * 4, 1, 4, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x)

class ImageEncoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Flatten(),
        )

        self.fc_mu = nn.Linear(256 * 8 * 8, latent_dim)
        self.fc_var = nn.Linear(256 * 8 * 8, latent_dim)

    def forward(self, x):
        features = self.encoder(x)
        return self.fc_mu(features), self.fc_var(features)

class TextEncoder(nn.Module):
    def __init__(self, latent_dim, num_attributes):
        super().__init__()
        hidden_dim = 512
        self.fc = nn.Sequential(
            nn.Linear(num_attributes, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim*2)
        )
        self.mu_head = nn.Linear(latent_dim*2, latent_dim)
        self.logvar_head = nn.Linear(latent_dim*2, latent_dim)

    def forward(self, attributes):
        x = self.fc(attributes)
        mu = self.mu_head(x)
        logvar = self.logvar_head(x)
        return mu, logvar

class TextDecoder(nn.Module):
    def __init__(self, latent_dim, num_attributes):
        super().__init__()
        hidden_dim = 512
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_attributes),
            nn.Sigmoid()
        )

    def forward(self, z, threshold=0.5):
        attribute_probs = self.fc(z)
        predicted_attributes = (attribute_probs > threshold).float()
        return attribute_probs, predicted_attributes

class ImageDecoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()

        self.decoder_input = nn.Sequential(
            nn.Linear(latent_dim, 2048),
            nn.LeakyReLU(0.2),
            ResidualLinear(2048),
            nn.Linear(2048, 512 * 4 * 4),
            nn.LeakyReLU(0.2)
        )

        self.attention = nn.MultiheadAttention(embed_dim=512, num_heads=8)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            ResidualBlock(512),

            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            ResidualBlock(256),

            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            ResidualBlock(128),

            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            ResidualBlock(64),

            nn.Conv2d(64, 3, 3, 1, 1),
            nn.Tanh()
        )

    def forward(self, z):
        x = self.decoder_input(z)
        x = x.view(-1,512,4,4)

        batch_size = x.size(0)
        x_flat = x.view(batch_size,512,-1).permute(2,0,1)
        attn_out, _ = self.attention(x_flat,x_flat,x_flat)
        x_flat = x_flat + attn_out
        x = x_flat.permute(1,2,0).view(batch_size,512,4,4)

        return self.decoder(x)

class MultimodalVAE(nn.Module):
    def __init__(self, latent_dim=512, num_attributes=10, temperature=1.0):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_attributes = num_attributes
        self.temperature = temperature

        self.image_encoder = ImageEncoder(latent_dim)
        self.text_encoder = TextEncoder(latent_dim, num_attributes)
        self.image_decoder = ImageDecoder(latent_dim)
        self.text_decoder = TextDecoder(latent_dim, num_attributes)

        self.norm_layer = nn.LayerNorm(latent_dim)

        self.idx_to_attribute = {
            0: 'young',
            1: 'male',
            2: 'female',
            3: 'smiling',
            4: 'eyeglasses',
            5: 'black_hair',
            6: 'blond_hair',
            7: 'bald',
            8: 'mustache',
            9: 'wearing_lipstick'
        }

        self.vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features[:16].eval().to(next(self.parameters()).device)
        for param in self.vgg.parameters():
            param.requires_grad = False

    def _attributes_to_text(self, predicted_attributes):
        batch_descriptions = []
        for attr_vector in predicted_attributes:
            active_attributes = [
                self.idx_to_attribute[idx].replace('_',' ')
                for idx, is_active in enumerate(attr_vector)
                if is_active == 1
            ]
            description = generate_natural_description(active_attributes)
            batch_descriptions.append(description)
        return batch_descriptions

    def reparameterize(self, mu, log_var):
        if self.training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            return mu + eps * std * self.temperature
        return mu

    def encode_image(self, images):
        mu, log_var = self.image_encoder(images)
        z = self.reparameterize(mu, log_var)
        return z, mu, log_var

    def encode_text(self, attributes):
        mu, log_var = self.text_encoder(attributes)
        z = self.reparameterize(mu, log_var)
        return z, mu, log_var

    def decode_image(self, z):
        return self.image_decoder(z)

    def decode_text(self, z, generation_params=None):
        if generation_params is None:
            generation_params = {'threshold':0.5}
        attr_probs, pred_attributes = self.text_decoder(z, **generation_params)
        descriptions = self._attributes_to_text(pred_attributes)
        return attr_probs, pred_attributes, descriptions

    def forward(self, images=None, target_attributes=None):
        outputs = {}

        if images is not None:
            z_image, image_mu, image_log_var = self.encode_image(images)
            outputs['image_mu'] = image_mu
            outputs['image_log_var'] = image_log_var
            outputs['z_image'] = z_image
            outputs['recon_images'] = self.decode_image(z_image)

            attr_probs_img, pred_attrs_img = self.text_decoder(z_image)
            outputs['text_from_image_probs'] = attr_probs_img

        if target_attributes is not None:
            z_text, text_mu, text_log_var = self.encode_text(target_attributes)
            outputs['text_mu'] = text_mu
            outputs['text_log_var'] = text_log_var
            outputs['z_text'] = z_text

            attr_probs, pred_attributes = self.text_decoder(z_text)
            outputs['recon_text_probs'] = attr_probs
            outputs['recon_text_attributes'] = pred_attributes
            outputs['recon_text'] = self._attributes_to_text(pred_attributes)

            image_from_text = self.decode_image(z_text)
            outputs['image_from_text'] = image_from_text

        if images is not None and target_attributes is not None:
            z_image_norm = F.normalize(self.norm_layer(outputs['z_image']), dim=-1)
            z_text_norm = F.normalize(self.norm_layer(outputs['z_text']), dim=-1)
            outputs['consistency_score'] = (F.cosine_similarity(z_image_norm, z_text_norm) + 1)/2

        return outputs

    def check_consistency(self, z1, z2):
        z1_norm = F.normalize(self.norm_layer(z1), dim=-1)
        z2_norm = F.normalize(self.norm_layer(z2), dim=-1)
        return (F.cosine_similarity(z1_norm, z2_norm) + 1)/2

    @torch.no_grad()
    def generate_from_text(self, attributes):
        device = next(self.parameters()).device
        attributes = attributes.to(device)
        z_text, _, _ = self.encode_text(attributes)
        return self.decode_image(z_text)

    @torch.no_grad()
    def generate_from_image(self, image):
        device = next(self.parameters()).device
        image = image.to(device)
        if image.dim() == 3:
            image = image.unsqueeze(0)
        z_image, _, _ = self.encode_image(image)
        _, pred_attributes = self.text_decoder(z_image)
        descriptions = self._attributes_to_text(pred_attributes)
        if len(descriptions) == 1:
            return descriptions[0]
        return descriptions

    def sample_latent(self, batch_size=1):
        device = next(self.parameters()).device
        return torch.randn(batch_size, self.latent_dim).to(device)

    def interpolate_latent(self, z1, z2, steps=10):
        alphas = torch.linspace(0,1,steps,device=z1.device)
        z_interp = torch.zeros(steps,self.latent_dim,device=z1.device)
        for i,alpha in enumerate(alphas):
            z_interp[i] = (1-alpha)*z1 + alpha*z2
        return z_interp

    def fuse_representations(self, image, target_attributes, fusion_weight=0.5):
        z_image, image_mu, _ = self.encode_image(image)
        z_text, text_mu, _ = self.encode_text(target_attributes)
        
        z_image_norm = self.norm_layer(z_image)
        z_text_norm = self.norm_layer(z_text)
        
        attention = torch.sigmoid(torch.sum(z_image_norm * z_text_norm, dim=-1, keepdim=True))
        
        z_fused = (1 - fusion_weight) * (z_image_norm + attention * z_image_norm) + \
                  fusion_weight * (z_text_norm + (1 - attention) * z_text_norm)
        
        z_fused = z_fused + 0.1 * z_image_norm
        
        return self.decode_image(z_fused)