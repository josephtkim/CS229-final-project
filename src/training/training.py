import os
import json
from datetime import datetime
import pandas as pd
import torch
import torch.nn.functional as F
from collections import defaultdict
from tqdm import tqdm

from .losses import (
    discriminator_loss, generator_loss, generator_loss_eval,
    discriminator_loss_eval, compute_distribution_matching_loss,
    identity_consistency_loss
)
from ..visualization.viz import visualize_results

class TrainingManager:
    def __init__(self, save_dir='training'):
        self.save_dir = save_dir
        os.makedirs(save_dir,exist_ok=True)
        self.history_path = os.path.join(save_dir,'history.csv')
        self.history=[]
        self.info_path = os.path.join(save_dir,'run_info.json')

    def save_checkpoint(self, epoch, model, discriminator, optimizer_G, optimizer_D, loss, phase):
        phase_checkpoint_dir = os.path.join(self.save_dir, 'checkpoints', f'phase{phase}')
        os.makedirs(phase_checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(phase_checkpoint_dir,f'checkpoint_epoch_{epoch}.pt')
        torch.save({
            'epoch':epoch,
            'model_state_dict':model.state_dict(),
            'discriminator_state_dict':discriminator.state_dict(),
            'optimizer_G_state_dict':optimizer_G.state_dict(),
            'optimizer_D_state_dict':optimizer_D.state_dict(),
            'loss':loss
        }, checkpoint_path)
        self._save_run_info(epoch+1,phase)

    def _save_run_info(self, last_epoch, phase):
        info = {
            'last_epoch':last_epoch,
            'phase':phase,
            'timestamp':datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        with open(self.info_path,'w') as f:
            json.dump(info,f)

    def log_metrics(self,epoch,metrics):
        metrics['epoch'] = epoch+1
        metrics['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.history.append(metrics)
        df = pd.DataFrame(self.history)
        df.to_csv(self.history_path,index=False)

def train_step(model, discriminator, batch, optimizer_G, optimizer_D, device, epoch, phase_config):
    images = batch['image'].to(device)
    target_attributes = batch['attributes'].to(device) if 'attributes' in batch else None

    outputs = model(images=images, target_attributes=target_attributes)

    losses = {}

    if phase_config['adversarial_weight'] > 0:
        optimizer_D.zero_grad()
        d_loss = phase_config['adversarial_weight'] * discriminator_loss(
            discriminator, images, outputs['recon_images'], epoch, phase_config
        )
        d_loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
        optimizer_D.step()
        losses['d_loss'] = d_loss.item()
    else:
        losses['d_loss'] = 0.0

    if phase_config['adversarial_weight'] > 0:
        losses['adversarial'] = phase_config['adversarial_weight'] * generator_loss(
            discriminator, outputs['recon_images']
        )
    else:
        losses['adversarial'] = torch.tensor(0.0, device=device)

    optimizer_G.zero_grad()

    # Use model's vgg for perceptual loss
    with torch.no_grad():
        target_feats = model.vgg(images)
    recon_feats = model.vgg(outputs['recon_images'])
    mse_loss = F.mse_loss(outputs['recon_images'], images)
    perceptual_loss_val = F.mse_loss(recon_feats, target_feats)
    identity_loss = identity_consistency_loss(
        model.image_encoder(images)[0],
        model.image_encoder(outputs['recon_images'])[0]
    )
    losses['image_recon'] = phase_config['reconstruction_weight']*(mse_loss + 0.05*perceptual_loss_val + 0.1*identity_loss)

    if target_attributes is not None:
        bce_loss = F.binary_cross_entropy(
            outputs['recon_text_probs'],
            target_attributes,
            reduction='none'
        )

        penalized_loss = torch.where(
            bce_loss>0.15,
            bce_loss*8.0,
            bce_loss
        )

        text_attr_loss = F.binary_cross_entropy(
            outputs['recon_text_probs'],
            target_attributes
        )

        losses['text_recon_loss'] = phase_config['text_reconstruction_weight'] * text_attr_loss

        img2text_attr_loss = F.binary_cross_entropy(
            outputs['text_from_image_probs'],
            target_attributes
        )
        losses['text_from_image'] = phase_config['cross_modal_weight'] * img2text_attr_loss

    attr_consistency_loss = F.mse_loss(
        outputs['recon_text_probs'],
        outputs['text_from_image_probs']
    )
    losses['attr_consistency'] = phase_config['consistency_weight']*attr_consistency_loss

    if 'image_from_text' in outputs:
        with torch.no_grad():
            target_feats_txt = model.vgg(images)
        recon_feats_txt = model.vgg(outputs['image_from_text'])
        losses['image_from_text'] = phase_config['cross_modal_weight']*(
            F.mse_loss(outputs['image_from_text'],images)+
            0.1*F.mse_loss(recon_feats_txt, target_feats_txt)+
            0.1*identity_consistency_loss(
                model.image_encoder(images)[0],
                model.image_encoder(outputs['image_from_text'])[0]
            )
        )

    if all(k in outputs for k in ['image_mu','image_log_var']):
        losses['image_kl'] = phase_config['kl_weight'] * (-0.5 * torch.mean(
            1+outputs['image_log_var'] - outputs['image_mu'].pow(2)-outputs['image_log_var'].exp()
        ))

    if all(k in outputs for k in ['text_mu','text_log_var']):
        losses['text_kl'] = phase_config['kl_weight'] * (-0.5*torch.mean(
            1+outputs['text_log_var']-outputs['text_mu'].pow(2)-outputs['text_log_var'].exp()
        ))

    if all(k in outputs for k in ['image_mu','image_log_var','text_mu','text_log_var']):
        losses['distribution_matching'] = phase_config['consistency_weight'] * compute_distribution_matching_loss(
            outputs['image_mu'], outputs['image_log_var'],
            outputs['text_mu'], outputs['text_log_var']
        )

    if 'consistency_score' in outputs:
        losses['consistency'] = phase_config['consistency_weight']*(1-outputs['consistency_score'].mean())

    if target_attributes is not None and phase_config['attribute_weight']>0:
        if 'image_attributes' in outputs:
            losses['image_attribute_loss'] = phase_config['attribute_weight'] * F.binary_cross_entropy_with_logits(
                outputs['image_attributes'],
                target_attributes
            )

    if target_attributes is not None and phase_config['attribute_weight']>0:
        if 'text_attributes' in outputs:
            losses['text_attribute_loss'] = phase_config['attribute_weight']*F.binary_cross_entropy_with_logits(
                outputs['text_attributes'],
                target_attributes
            )

    total_loss = sum(losses.values())
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer_G.step()

    losses['total_loss'] = total_loss.item()
    return {k:(v.item() if isinstance(v,torch.Tensor) else v) for k,v in losses.items()}

def validate(model, discriminator, val_loader, device, epoch, phase_config):
    model.eval()
    discriminator.eval()
    metrics = {
        'val_image_recon_loss':0,
        'val_text_recon_loss':0,
        'val_text_from_image_loss':0,
        'val_image_from_text_loss':0,
        'val_image_kl_loss':0,
        'val_text_kl_loss':0,
        'val_distribution_matching_loss':0,
        'val_consistency_loss':0,
        'val_adversarial_loss':0,
        'val_d_loss':0,
        'val_attribute_accuracy':0,
        'val_key_attribute_accuracy':0,
        'val_total_loss':0
    }

    num_batches=0
    with torch.no_grad():
        for batch in val_loader:
            num_batches+=1
            images = batch['image'].to(device)
            target_attributes = batch['attributes'].to(device)

            outputs = model(images=images, target_attributes=target_attributes)

            with torch.no_grad():
                target_feats = model.vgg(images)
            recon_feats = model.vgg(outputs['recon_images'])

            mse_loss = F.mse_loss(outputs['recon_images'], images)
            perceptual_loss_val = F.mse_loss(recon_feats, target_feats)
            identity_loss = identity_consistency_loss(
                model.image_encoder(images)[0],
                model.image_encoder(outputs['recon_images'])[0]
            )
            metrics['val_image_recon_loss'] += phase_config['reconstruction_weight']*(mse_loss+0.05*perceptual_loss_val+0.1*identity_loss).item()

            text_attr_loss = F.binary_cross_entropy(
                outputs['recon_text_probs'],
                target_attributes
            )
            metrics['val_text_recon_loss'] = phase_config['text_reconstruction_weight'] * text_attr_loss

            pred_attributes = (outputs['recon_text_probs']>0.5).float()
            accuracy = (pred_attributes == target_attributes).float().mean()
            metrics['val_attribute_accuracy'] += accuracy.item()

            if 'text_from_image_probs' in outputs:
                img2text_attr_loss = F.binary_cross_entropy(
                    outputs['text_from_image_probs'],
                    target_attributes
                )
                metrics['val_text_from_image_loss'] += (phase_config['cross_modal_weight']*img2text_attr_loss).item()

            if 'image_from_text' in outputs:
                txt_target_feats = model.vgg(images)
                txt_recon_feats = model.vgg(outputs['image_from_text'])
                image_from_text_loss = phase_config['cross_modal_weight'] * (
                    F.mse_loss(outputs['image_from_text'],images) +
                    0.1 * F.mse_loss(txt_recon_feats, txt_target_feats) +
                    0.1 * identity_consistency_loss(
                        model.image_encoder(images)[0],
                        model.image_encoder(outputs['image_from_text'])[0]
                    )
                )
                metrics['val_image_from_text_loss'] += image_from_text_loss.item()

            if all(k in outputs for k in ['image_mu','image_log_var']):
                image_kl = -0.5*torch.mean(
                    1+outputs['image_log_var']-outputs['image_mu'].pow(2)-outputs['image_log_var'].exp()
                )
                metrics['val_image_kl_loss'] += (phase_config['kl_weight']*image_kl).item()

            if all(k in outputs for k in ['text_mu','text_log_var']):
                text_kl = -0.5*torch.mean(
                    1+outputs['text_log_var']-outputs['text_mu'].pow(2)-outputs['text_log_var'].exp()
                )
                metrics['val_text_kl_loss'] += (phase_config['kl_weight']*text_kl).item()

            if all(k in outputs for k in ['image_mu','image_log_var','text_mu','text_log_var']):
                matching_loss = compute_distribution_matching_loss(
                    outputs['image_mu'], outputs['image_log_var'],
                    outputs['text_mu'], outputs['text_log_var']
                )
                metrics['val_distribution_matching_loss'] += (phase_config['consistency_weight']*matching_loss).item()

            if 'consistency_score' in outputs:
                consistency_loss = 1-outputs['consistency_score'].mean()
                metrics['val_consistency_loss'] += (phase_config['consistency_weight']*consistency_loss).item()

            if phase_config['adversarial_weight']>0:
                d_loss = discriminator_loss_eval(discriminator, images, outputs['recon_images'], phase_config)
                metrics['val_d_loss'] += d_loss.item()

                adv_loss = phase_config['adversarial_weight']*generator_loss_eval(discriminator, outputs['recon_images'])
                metrics['val_adversarial_loss'] += adv_loss.item()

    metrics = {k:v/num_batches for k,v in metrics.items()}
    metrics['val_total_loss'] = sum(v for k,v in metrics.items() if k not in ['val_attribute_accuracy','val_key_attribute_accuracy'])
    return metrics

def evaluate_model(model, discriminator, test_loader, device, epoch, phase_config):
    model.eval()
    metrics = validate(model, discriminator, test_loader, device, epoch, phase_config)

    with torch.no_grad():
        for batch in test_loader:
            images = batch['image'].to(device)
            target_attributes = batch['attributes'].to(device)
            outputs = model(images=images, target_attributes=target_attributes)
            pred_attributes = (outputs['text_from_image_probs']>0.5).float()
            per_attr_acc = (pred_attributes == target_attributes).float().mean(dim=0)
            print("\nPer-Attribute Accuracy:")
            for idx, acc in enumerate(per_attr_acc):
                attr_name = model.idx_to_attribute[idx].replace('_',' ')
                print(f"{attr_name}: {acc.item():.4f}")
            break
    return metrics

def train_phase_1(model, discriminator, train_loader, val_loader, optimizer_G, optimizer_D, device, num_epochs, trainer, config, val_subset, start_epoch):
    print("\nStarting Phase 1: Early Training")
    phase_config = config['phase_configs'][1]
    best_val_loss = float('inf')
    best_epoch=0

    if num_epochs>0:
        for epoch in range(start_epoch,num_epochs):
            model.train()
            discriminator.train()
            train_metrics=defaultdict(float)
            for batch in tqdm(train_loader, desc=f"Phase 1 - Epoch {epoch+1}/{num_epochs}"):
                metrics = train_step(model,discriminator,batch,optimizer_G,optimizer_D,device,epoch,phase_config)
                for k,v in metrics.items():
                    train_metrics[k]+=v
            train_metrics={k:v/len(train_loader) for k,v in train_metrics.items()}

            val_metrics=validate(model,discriminator,val_loader,device,epoch,phase_config)
            if val_metrics['val_total_loss']<best_val_loss:
                best_val_loss=val_metrics['val_total_loss']
                best_epoch=epoch

            if (epoch+1)%config['eval_freq']==0:
                eval_metrics = evaluate_model(model,discriminator,val_loader,device,epoch+1,phase_config)
                from visualization import visualize_results
                results_dir = visualize_results(model,val_subset,epoch+1,"phase1",num_samples=config['num_vis_samples'],device=device)
                print(f"Phase 1 - Visualization results saved to {results_dir}")

            combined_metrics = {
                'epoch':epoch+1,
                'phase':1,
                'phase_epoch':epoch+1,
                'total_epochs':num_epochs,
                **train_metrics,
                **val_metrics
            }
            trainer.log_metrics(epoch,combined_metrics)

            print(f"\nPhase 1 - Epoch {epoch+1} Metrics:")
            print("\nTraining Metrics:")
            for k,v in train_metrics.items():
                print(f"  {k}: {v:.4f}")

            print("\nValidation Metrics:")
            for k,v in val_metrics.items():
                print(f"  {k}: {v:.4f}")

        trainer.save_checkpoint(epoch=num_epochs,model=model,discriminator=discriminator,optimizer_G=optimizer_G,optimizer_D=optimizer_D,loss=val_metrics['val_total_loss'],phase=1)
    return best_val_loss,best_epoch

def train_phase_2(model, discriminator, train_loader, val_loader, optimizer_G, optimizer_D, device, num_epochs, trainer, config, val_subset, start_epoch):
    print("\nStarting Phase 2: Middle Training")
    phase_config = config['phase_configs'][2]
    best_val_loss=float('inf')
    best_epoch=0

    if num_epochs>0:
        for epoch in range(start_epoch,num_epochs):
            model.train()
            discriminator.train()

            train_metrics=defaultdict(float)
            for batch in tqdm(train_loader, desc=f"Phase 2 - Epoch {epoch+1}/{num_epochs}"):
                metrics = train_step(model,discriminator,batch,optimizer_G,optimizer_D,device,epoch,phase_config)
                for k,v in metrics.items():
                    train_metrics[k]+=v
            train_metrics={k:v/len(train_loader) for k,v in train_metrics.items()}

            val_metrics=validate(model,discriminator,val_loader,device,epoch,phase_config)
            if val_metrics['val_total_loss']<best_val_loss:
                best_val_loss=val_metrics['val_total_loss']
                best_epoch=epoch

            if (epoch+1)%config['eval_freq']==0:
                eval_metrics=evaluate_model(model,discriminator,val_loader,device,epoch+1,phase_config)
                from visualization import visualize_results
                results_dir = visualize_results(model,val_subset,epoch+1,"phase2",num_samples=config['num_vis_samples'],device=device)
                print(f"Phase 2 - Visualization results saved to {results_dir}")

            combined_metrics={
                'epoch':epoch+1,
                'phase':2,
                'phase_epoch':epoch+1,
                'total_epochs':num_epochs,
                **train_metrics,
                **val_metrics
            }
            trainer.log_metrics(epoch,combined_metrics)

            print(f"\nPhase 2 - Epoch {epoch+1} Metrics:")
            print("\nTraining Metrics:")
            for k,v in train_metrics.items():
                print(f"  {k}: {v:.4f}")

            print("\nValidation Metrics:")
            for k,v in val_metrics.items():
                print(f"  {k}: {v:.4f}")

        trainer.save_checkpoint(epoch=num_epochs,model=model,discriminator=discriminator,optimizer_G=optimizer_G,optimizer_D=optimizer_D,loss=val_metrics['val_total_loss'],phase=2)
    return best_val_loss,best_epoch

def train_phase_3(model, discriminator, train_loader, val_loader, optimizer_G, optimizer_D, device, num_epochs, trainer, config, val_subset, start_epoch):
    print("\nStarting Phase 3: Late Training")
    phase_config = config['phase_configs'][3]
    best_val_loss=float('inf')
    best_epoch=0

    if num_epochs>0:
        for epoch in range(start_epoch,num_epochs):
            model.train()
            discriminator.train()

            train_metrics=defaultdict(float)
            for batch in tqdm(train_loader, desc=f"Phase 3 - Epoch {epoch+1}/{num_epochs}"):
                metrics = train_step(model,discriminator,batch,optimizer_G,optimizer_D,device,epoch,phase_config)
                for k,v in metrics.items():
                    train_metrics[k]+=v
            train_metrics={k:v/len(train_loader) for k,v in train_metrics.items()}

            val_metrics=validate(model,discriminator,val_loader,device,epoch,phase_config)
            if val_metrics['val_total_loss']<best_val_loss:
                best_val_loss=val_metrics['val_total_loss']
                best_epoch=epoch

            if (epoch+1)%config['eval_freq']==0:
                eval_metrics = evaluate_model(model,discriminator,val_loader,device,epoch+1,phase_config)
                from visualization import visualize_results
                results_dir=visualize_results(model,val_subset,epoch+1,"phase3",num_samples=config['num_vis_samples'],device=device)
                print(f"Phase 3 - Visualization results saved to {results_dir}")

            combined_metrics={
                'epoch':epoch+1,
                'phase':3,
                'phase_epoch':epoch+1,
                'total_epochs':num_epochs,
                **train_metrics,
                **val_metrics
            }
            trainer.log_metrics(epoch,combined_metrics)

            print(f"\nPhase 3 - Epoch {epoch+1} Metrics:")
            print("\nTraining Metrics:")
            for k,v in train_metrics.items():
                print(f"  {k}: {v:.4f}")

            print("\nValidation Metrics:")
            for k,v in val_metrics.items():
                print(f"  {k}: {v:.4f}")

        trainer.save_checkpoint(epoch=num_epochs,model=model,discriminator=discriminator,optimizer_G=optimizer_G,optimizer_D=optimizer_D,loss=val_metrics['val_total_loss'],phase=3)
    return best_val_loss,best_epoch

def run_phased_training(model, discriminator, train_loader, val_loader, optimizer_G, optimizer_D,
                       device, config, trainer, val_subset, phase1_epochs, phase2_epochs, phase3_epochs,
                       phase1_start=0, phase2_start=0, phase3_start=0):

    total_epochs=phase1_epochs+phase2_epochs+phase3_epochs
    print(f"\nStarting phased training:")
    print(f"Phase 1: {phase1_epochs} epochs")
    print(f"Phase 2: {phase2_epochs} epochs")
    print(f"Phase 3: {phase3_epochs} epochs")
    print(f"Total: {total_epochs} epochs")

    best_losses=[]

    if phase1_epochs>0:
        print(f"\nPhase 1: Early Training (Reconstruction Focus)")
        phase1_loss,phase1_best = train_phase_1(
            model,discriminator,train_loader,val_loader,optimizer_G,optimizer_D,
            device,phase1_epochs,trainer,config,val_subset,phase1_start
        )
        print(f"\nPhase 1 completed. Best loss: {phase1_loss:.4f} at epoch {phase1_best+1}")
        best_losses.append(phase1_loss)

    if phase2_epochs>0:
        print(f"\nPhase 2: Middle Training (Alignment Focus)")
        phase2_loss,phase2_best = train_phase_2(
            model,discriminator,train_loader,val_loader,optimizer_G,optimizer_D,
            device,phase2_epochs,trainer,config,val_subset,phase2_start
        )
        print(f"\nPhase 2 completed. Best loss: {phase2_loss:.4f} at epoch {phase2_best+1}")
        best_losses.append(phase2_loss)

    if phase3_epochs>0:
        print(f"\nPhase 3: Late Training (Refinement Focus)")
        phase3_loss,phase3_best = train_phase_3(
            model,discriminator,train_loader,val_loader,optimizer_G,optimizer_D,
            device,phase3_epochs,trainer,config,val_subset,phase3_start
        )
        print(f"\nPhase 3 completed. Best loss: {phase3_loss:.4f} at epoch {phase3_best+1}")
        best_losses.append(phase3_loss)

    return min(best_losses) if best_losses else float('inf')
