import os
import matplotlib.pyplot as plt
import textwrap
import torch
import re
from utils import clean_and_validate_attributes, generate_natural_description

class VisualizationUtils:
    def __init__(self, save_dir='results'):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def _clean_text(self, text):
        text = re.sub(r'[^\x00-\x7F]+','',text)
        text = re.sub(r'\s+',' ',text)
        return text.strip()

    def save_face_grid(self, images, captions, epoch, filename):
        num_images = len(images)
        num_cols = min(5,num_images)
        num_rows = (num_images+num_cols-1)//num_cols

        fig = plt.figure(figsize=(4*num_cols,5*num_rows))
        for idx,(image,caption) in enumerate(zip(images,captions)):
            ax = fig.add_subplot(num_rows,num_cols,idx+1)
            if isinstance(image,torch.Tensor):
                image = (image.detach().cpu().clone()+1)/2
                image = image.permute(1,2,0).numpy()
            ax.imshow(image)
            ax.axis('off')
            cleaned_caption = self._clean_text(caption)
            wrapped_caption = '\n'.join(textwrap.wrap(cleaned_caption,width=40))
            ax.set_title(wrapped_caption,fontsize=8,pad=10)

        plt.tight_layout(pad=3.0)
        plt.savefig(os.path.join(self.save_dir, filename), bbox_inches='tight')
        plt.close()

    def save_face_comparison(self, real_images, generated_images, captions, epoch):
        num_samples = len(real_images)
        fig, axes = plt.subplots(num_samples, 2, figsize=(8,5*num_samples))
        for i in range(num_samples):
            real_img = (real_images[i].detach().cpu().clone()+1)/2
            axes[i,0].imshow(real_img.permute(1,2,0).numpy())
            axes[i,0].set_title("Real",pad=10)
            axes[i,0].axis('off')

            gen_img = (generated_images[i].detach().cpu().clone()+1)/2
            axes[i,1].imshow(gen_img.permute(1,2,0).numpy())
            axes[i,1].set_title("Generated",pad=10)
            axes[i,1].axis('off')

        plt.tight_layout(pad=3.0, rect=[0,0,1,0.95])
        plt.savefig(os.path.join(self.save_dir, f'face_comparison_epoch_{epoch}.png'), bbox_inches='tight')
        plt.close()

    def save_attribute_manipulation(self, base_image, manipulated_images, attributes, epoch):
        num_variations = len(manipulated_images)
        fig, axes = plt.subplots(1, num_variations+1, figsize=(4*(num_variations+1),5))
        fig.subplots_adjust(bottom=0.2)

        def process_image(img_tensor):
            img = (img_tensor.detach().cpu().clone()+1)/2
            img = torch.clamp(img,0,1)
            return img.permute(1,2,0).numpy()

        base_img = process_image(base_image)
        ax_base = axes[0]
        ax_base.imshow(base_img)
        ax_base.set_title("Original",pad=20)
        ax_base.axis('off')
        ax_base.set_xlim(-base_img.shape[1]*0.1, base_img.shape[1]*1.1)
        ax_base.set_ylim(base_img.shape[0]*1.1, -base_img.shape[0]*0.1)

        for i,(img,attr) in enumerate(zip(manipulated_images,attributes)):
            img = process_image(img)
            ax = axes[i+1]
            ax.imshow(img)
            ax.set_title(attr, pad=20)
            ax.axis('off')
            ax.set_xlim(-img.shape[1]*0.1,img.shape[1]*1.1)
            ax.set_ylim(img.shape[0]*1.1,-img.shape[0]*0.1)

        plt.tight_layout()
        save_path = os.path.join(self.save_dir, f'attribute_manipulation_epoch_{epoch}.png')
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()

    def save_text_recon_results(self, input_texts, recon_texts, epoch):
        file_path = os.path.join(self.save_dir, f'txt2txt_epoch_{epoch}.txt')
        with open(file_path,'w',encoding='utf-8') as f:
            for i,(input_text,recon_text) in enumerate(zip(input_texts,recon_texts)):
                f.write(f"Sample {i+1}:\n")
                f.write(f"Input: {input_text}\n")
                f.write(f"Reconstructed: {recon_text}\n\n")

    def save_img2text_results(self, input_images, generated_texts, original_captions, epoch):
        fig, axes = plt.subplots(3,5,figsize=(20,12),gridspec_kw={'height_ratios':[3,1,1]})
        fig.suptitle(f'Image to Text Generation - Epoch {epoch}', y=1.05)

        for i in range(5):
            if i<len(input_images):
                img = (input_images[i].cpu().clone()+1)/2
                img = img.permute(1,2,0)

                axes[0][i].imshow(img)
                axes[0][i].axis('off')

                generated_text = generated_texts[i] if i<len(generated_texts) else "No text generated"
                generated_text = self._clean_text(generated_text)
                wrapped_generated_text = '\n'.join(textwrap.wrap(generated_text,width=30))
                axes[1][i].text(0.5,0.5,f"Generated: {wrapped_generated_text}",ha='center',va='center',wrap=True,fontsize=8)
                axes[1][i].axis('off')

                original_caption = original_captions[i] if i<len(original_captions) else "No caption available"
                original_caption = self._clean_text(original_caption)
                wrapped_original_caption = '\n'.join(textwrap.wrap(original_caption,width=30))
                axes[2][i].text(0.5,0.5,f"Original: {wrapped_original_caption}",ha='center',va='center',wrap=True,fontsize=8)
                axes[2][i].axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, f'img2text_epoch_{epoch}.png'))
        plt.close()

    def save_text2img_results(self, generated_images, input_texts, epoch):
        fig, axes = plt.subplots(1,5,figsize=(20,4))
        fig.suptitle(f'Text to Image Generation - Epoch {epoch}', y=1.05)
        for i in range(5):
            if i<len(generated_images):
                img = (generated_images[i].cpu().clone()+1)/2
                img = img.permute(1,2,0)
                axes[i].imshow(img)
                axes[i].axis('off')
                cleaned_text = self._clean_text(input_texts[i])
                wrapped_text = '\n'.join(textwrap.wrap(cleaned_text,width=20))
                axes[i].set_title(wrapped_text,fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, f'text2img_epoch_{epoch}.png'))
        plt.close()

    def save_img2img_results(self, input_images, recon_images, epoch):
        num_samples = len(input_images)
        fig, axes = plt.subplots(2,num_samples,figsize=(4*num_samples,8))
        fig.suptitle('Original (top) vs Reconstructed (bottom) Faces',y=0.95)
        for i in range(num_samples):
            input_img = (input_images[i].cpu().clone()+1)/2
            axes[0,i].imshow(input_img.permute(1,2,0))
            axes[0,i].set_title("Original",pad=10)
            axes[0,i].axis('off')

            recon_img = (recon_images[i].detach().cpu().clone()+1)/2
            axes[1,i].imshow(recon_img.permute(1,2,0))
            axes[1,i].set_title("Reconstructed",pad=10)
            axes[1,i].axis('off')

        plt.tight_layout(pad=3.0,rect=[0,0,1,0.95])
        plt.savefig(os.path.join(self.save_dir, f'img2img_recon_epoch_{epoch}.png'), bbox_inches='tight')
        plt.close()

    def save_comparisons(self, samples, epoch):
        fig = plt.figure(figsize=(20,20))
        for idx in range(min(4,len(samples))):
            original_image = samples[idx]['original_image']
            generated_image = samples[idx]['generated_image']
            original_text = samples[idx]['original_text']
            generated_text = samples[idx]['generated_text']

            base_idx = idx*4
            row_start = (idx//2)*2
            col_start = (idx%2)*2

            ax1 = plt.subplot2grid((8,4),(row_start, col_start))
            orig_img = (original_image.cpu().clone()+1)/2
            ax1.imshow(orig_img.permute(1,2,0))
            ax1.set_title('Original Image')
            ax1.axis('off')

            ax2 = plt.subplot2grid((8,4),(row_start, col_start+1))
            gen_img = (generated_image.cpu().clone()+1)/2
            ax2.imshow(gen_img.permute(1,2,0))
            ax2.set_title('Generated Image')
            ax2.axis('off')

            ax3 = plt.subplot2grid((8,4),(row_start+1, col_start))
            cleaned_orig_text = self._clean_text(original_text)
            wrapped_orig_text = '\n'.join(textwrap.wrap(cleaned_orig_text,width=30))
            ax3.text(0.5,0.5,f'Original Text:\n{wrapped_orig_text}',ha='center',va='center',wrap=True)
            ax3.axis('off')

            ax4 = plt.subplot2grid((8,4),(row_start+1, col_start+1))
            cleaned_gen_text = self._clean_text(generated_text)
            wrapped_gen_text = '\n'.join(textwrap.wrap(cleaned_gen_text,width=30))
            ax4.text(0.5,0.5,f'Generated Text:\n{wrapped_gen_text}',ha='center',va='center',wrap=True)
            ax4.axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir,f'comparisons_epoch_{epoch}.png'))
        plt.close()

def visualize_results(model, test_dataset, epoch, phase_name, num_samples=5, device='cuda'):
    viz = VisualizationUtils(save_dir=f'results/{phase_name}')
    samples = [test_dataset[i] for i in range(min(num_samples,len(test_dataset)))]
    images = torch.stack([s['image'] for s in samples]).to(device)
    texts = [s['caption'] for s in samples]

    with torch.no_grad():
        generated_texts = model.generate_from_image(images)
        if isinstance(generated_texts,str):
            generated_texts = [generated_texts]*len(images)
        elif not isinstance(generated_texts,list):
            generated_texts = generated_texts.split(" ")

        cleaned_generated_texts = []
        for text in generated_texts:
            cleaned_words = [word for word in text.split() if word.isalnum()]
            cleaned_text = generate_natural_description(cleaned_words)
            cleaned_generated_texts.append(cleaned_text)

        generated_texts = cleaned_generated_texts
        generated_images = []
        for text in texts:
            extracted_attrs = clean_and_validate_attributes(text.split())
            text_attrs = torch.zeros(model.num_attributes)
            for attr in extracted_attrs:
                attr_key = attr.lower().replace(' ','_')
                for k,v in model.idx_to_attribute.items():
                    if v == attr_key:
                        text_attrs[k]=1.0
            text_attrs = text_attrs.unsqueeze(0).to(device)
            generated_image = model.generate_from_text(text_attrs)
            generated_images.append(generated_image[0])
        generated_images = torch.stack(generated_images)

        reconstructed_images = model.decode_image(model.encode_image(images)[0])

        attributes = torch.stack([s['attributes'] for s in samples]).to(device)
        z_text, _, _ = model.encode_text(attributes)
        attr_probs, pred_attributes = model.text_decoder(z_text)
        recon_texts = model._attributes_to_text(pred_attributes)
        if isinstance(recon_texts,str):
            recon_texts = [recon_texts]

    viz.save_face_grid(images.detach().cpu(), texts, epoch, f'original_faces_epoch_{epoch}.png')
    viz.save_face_grid(generated_images.detach().cpu(), texts, epoch, f'txt2img_epoch_{epoch}.png')
    viz.save_face_grid(reconstructed_images.detach().cpu(), generated_texts, epoch, f'img2img_img2text_epoch_{epoch}.png')
    viz.save_face_comparison(images.detach().cpu(), generated_images.detach().cpu(), texts, epoch)

    viz.save_img2text_results(images.detach().cpu(), generated_texts, texts, epoch)
    viz.save_text_recon_results(input_texts=texts, recon_texts=recon_texts, epoch=epoch)

    if epoch % 10 == 0:
        base_image = images[0]
        attributes_to_add = ['young','smiling','glasses','blond hair']
        manipulated_images=[]
        for attr in attributes_to_add:
            extracted_attrs = clean_and_validate_attributes([attr])
            text_attrs = torch.zeros((1,model.num_attributes),device=device)
            for extracted_attr in extracted_attrs:
                attr_key = extracted_attr.lower().replace(' ','_')
                for idx,attribute_name in model.idx_to_attribute.items():
                    if attr_key == attribute_name:
                        text_attrs[0,idx] = 1.0
            with torch.no_grad():
                manipulated_image = model.generate_from_text(text_attrs)
            manipulated_images.append(manipulated_image[0].detach().cpu())

        viz.save_attribute_manipulation(base_image.detach().cpu(), manipulated_images, attributes_to_add, epoch)

    print(f"Results saved in {viz.save_dir}")
    return viz.save_dir
