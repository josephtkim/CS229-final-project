# Multimodal VAE for Text-Image Tasks

This repository contains the implementation of a multimodal variational autoencoder (VAE) architecture that performs both consistency checking and cross-modal generation between text and image modalities.

## Overview

The model consists of two VAEs (Text VAE and Image VAE) with a shared latent space, enabling:
- Consistency checking between text-image pairs
- Text-to-image generation
- Image-to-text generation

![Model Architecture](assets/architecture.png)
*Full architecture showing Text VAE and Image VAE components with cross-modal pathways and discriminator.*

## Results
- Achieved F1-score of 0.89 on consistency checking
- Qualitatively coherent cross-modal generation
- Effective shared latent space representation

### Consistency Checking Results
![Consistency Checking](assets/consistency_pairs.png)
*Examples of consistency checking between text-image pairs with their similarity scores.*

### Latent Space Visualization
![Latent Space Alignment](assets/latent_space.png)
*t-SNE visualization of image and text embeddings in the shared latent space, showing clear gender-based clustering and alignment between modalities.*

### Cross-Modal Generation Examples
![Text to Image Generation](assets/text_to_image.png)
*Text-to-image generation examples showing the model's ability to generate faces matching textual descriptions.*

![Image to Text Generation](assets/image_to_text.png)
*Image-to-text generation examples demonstrating accurate attribute extraction and description generation.*

## Dataset

We use the CelebAMask-HQ dataset (Lee et al., 2020) with the following specifications:
- 5,000 examples (split 90/10 for training/validation) selected from the full 30,000 high-resolution face images
- Images resized to 64x64 resolution (original 512x512)
- 10 binary attributes (young, male/female, smiling, eyeglasses, black hair, blond hair, bald, mustache, wearing lipstick)

![Dataset Examples](assets/dataset_examples.png)
*Examples from the CelebAMask-HQ dataset with corresponding text descriptions.*

The CelebAMask-HQ dataset is available for non-commercial research purposes only. For more details, visit the [CelebAMask-HQ project page](https://mmlab.ie.cuhk.edu.hk/projects/CelebA/CelebAMask_HQ.html).

## Model Architecture

### Text VAE
- Fully connected layers for encoder and decoder
- Binary attribute vector input
- Sigmoid activation for attribute probability output

### Image VAE
- Convolutional encoder and decoder networks
- Patch discriminator for image quality improvement
- ResidualBlocks and ResidualLinear components
- VGG16 for perceptual loss computation

## Training

The model uses a three-phase training strategy:

1. **Phase 1: Unimodal Training**
   - Focus on reconstruction in each modality
   - 40 epochs

2. **Phase 2: Cross-Modal Alignment**
   - Introduce consistency and distribution matching
   - 30 epochs

3. **Phase 3: Adversarial Refinement**
   - Add adversarial training for image quality
   - 70 epochs

### Hyperparameters
- Latent dimensions: 256 (also tested with 128 and 512)
- Batch size: 16
- Learning rate: 1e-4 (generator), 1e-5 (discriminator)
- Random seed: 42

## Citation

If you use this code in your research, please cite:
@article{kim2024exploring,
title={Exploring Effectiveness of Multimodal Variational Autoencoders on Text-Image Tasks},
author={Kim, Joseph Taewoo},
journal={Stanford University},
year={2024}
}

For the CelebAMask-HQ dataset, please cite:
@inproceedings{CelebAMask-HQ,
title={MaskGAN: Towards Diverse and Interactive Facial Image Manipulation},
author={Lee, Cheng-Han and Liu, Ziwei and Wu, Lingyun and Luo, Ping},
booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
year={2020}
}
