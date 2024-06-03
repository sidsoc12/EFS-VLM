# BEGIN ADDITION VisAttention Addition
import matplotlib.pyplot as plt
import numpy as np
from torchvision.transforms.functional import to_pil_image

def visualize_attention(image, attn_weights, layer_num, head=0):
    attn = attn_weights[layer_num][0][head].detach().cpu().numpy()
    
    # Assuming `attn` is of shape (num_text_tokens, num_latent_tokens)
    # We need to map `attn` back to the image space
    
    # Example: Assuming 8x8 patches (this will depend on your specific model setup)
    num_patches = 64  # 8x8
    patch_size = int(np.sqrt(attn.shape[1]))
    
    # Reshape attention weights to match the image dimensions
    attn = attn.reshape((num_patches, num_patches))
    
    # Resize attention map to match image size
    attn_resized = np.kron(attn, np.ones((patch_size, patch_size)))
    
    # Normalize the attention map for better visualization
    attn_resized = attn_resized / attn_resized.max()
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(image)
    ax.imshow(attn_resized, cmap='jet', alpha=0.5)  # Overlay attention heatmap
    plt.title(f"Attention Layer {layer_num} Head {head}")
    plt.axis('off')
    plt.show()
# END ADDITION (VisAttention Addition)
