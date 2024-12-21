config = {
    'latent_dim': 256,
    'batch_size': 16,
    'learning_rate_G': 1e-4,
    'learning_rate_D': 1e-5,
    'checkpoint_freq': 10,
    'eval_freq': 5,
    'num_workers': 4,
    'data_path': "./data/CelebAMask-HQ",
    'cache_path': "./data/celeba_dataset_cache.pkl",
    'num_vis_samples': 5,
    'val_split': 0.1,
    'seed': 42,
    'phase1_epochs': 40,
    'phase2_epochs': 30,
    'phase3_epochs': 70,
    'resume_phase': None,
    'resume_epoch': None,

    'phase_configs': {
        1: {
            'reconstruction_weight': 2.0,
            'text_reconstruction_weight': 1.0,
            'attribute_weight': 1.0,
            'consistency_weight': 0.0,
            'kl_weight': 0.0,
            'cross_modal_weight': 0.0,
            'adversarial_weight': 0.0,
            'instance_noise': 0.0,
            'noise_decay': 0.9,
            'use_gradient_penalty': False
        },
        2: {
            'reconstruction_weight': 1.0,
            'text_reconstruction_weight': 1.0,
            'attribute_weight': 1.0,
            'consistency_weight': 0.05,
            'kl_weight': 0.005,
            'cross_modal_weight': 0.1,
            'adversarial_weight': 0.0,
            'instance_noise': 0.0,
            'noise_decay': 0.9,
            'use_gradient_penalty': False
        },
        3: {
            'reconstruction_weight': 1.0,
            'text_reconstruction_weight': 1.0,
            'attribute_weight': 1.0,
            'consistency_weight': 0.2,
            'kl_weight': 0.01,
            'cross_modal_weight': 0.3,
            'adversarial_weight': 0.001,
            'instance_noise': 0.01,
            'noise_decay': 0.9,
            'use_gradient_penalty': True
        }
    }
}
