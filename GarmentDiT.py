import torch
from .src.transformer_sd3_garm import SD3Transformer2DModel
import safetensors.torch

class GarmentEnhancementNode:
    """
    A custom node for garment enhancement using a pre-trained DiT model.
    """

    def __init__(self):
        super().__init__()
        # Load the transformer model
        model_path = "/kaggle/working/ComfyUI/models/DiT"
        self.transformer = SD3Transformer2DModel.from_pretrained(
            model_path, torch_dtype=torch.float16, local_files_only=True
        )
        self.transformer.eval()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip_embedding1": ("CLIP_VISION_OUTPUT",),
                "clip_embedding2": ("CLIP_VISION_OUTPUT",),
                "timestep": ("INT",),
            },
        }

    RETURN_TYPES = ("LATENT",)  # Outputs enhanced latent for decoding
    RETURN_NAMES = ("latent",)
    FUNCTION = "enhance_garment"
    CATEGORY = "Custom/Garment Enhancement"

    def enhance_garment(self, clip_embedding1, clip_embedding2, timestep):
        """
        Enhance the latent space or latent encoding using the transformer model and CLIP embeddings.
        No latent_image_tensor is used since the CLIP embeddings directly guide the enhancement.
        """
        # Extract CLIP embeddings (this contains the visual feature information)
        clip_tensor1 = clip_embedding1.last_hidden_state.to(torch.float16)
        clip_tensor2 = clip_embedding2.last_hidden_state.to(torch.float16)

        # Debugging: Print tensor shapes
        print(f"clip_tensor1 shape: {clip_tensor1.shape}")
        print(f"clip_tensor2 shape: {clip_tensor2.shape}")

        # Align dimensions dynamically (handle all dimensions except dim=1)
        max_dim1 = max(clip_tensor1.size(1), clip_tensor2.size(1))
        max_dim2 = max(clip_tensor1.size(2), clip_tensor2.size(2))

        clip_tensor1 = torch.nn.functional.pad(
            clip_tensor1, 
            (0, max_dim2 - clip_tensor1.size(2), 0, max_dim1 - clip_tensor1.size(1))
        )
        clip_tensor2 = torch.nn.functional.pad(
            clip_tensor2, 
            (0, max_dim2 - clip_tensor2.size(2), 0, max_dim1 - clip_tensor2.size(1))
        )

        batch_size = clip_tensor.shape[0]
        in_channels = 16  # From config
        patch_size = 2    # From config
        sample_size = 128  # From config
        height, width = sample_size // patch_size, sample_size // patch_size

        # Reshape to (batch_size, in_channels, height, width)
        clip_tensor = clip_tensor.view(batch_size, in_channels, height, width)

        # Pass through transformer model
        with torch.no_grad():
            output = self.transformer(
                hidden_states=clip_tensor,
                timestep=torch.tensor([timestep], dtype=torch.long),
                pooled_projections=pooled_projections,
                return_dict=True,
            )

        # Return the enhanced latent representation
        return (output.sample,)


# Node registration
NODE_CLASS_MAPPINGS = {
    "GarmentEnhancementNode": GarmentEnhancementNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GarmentEnhancementNode": "Garment Enhancement"
}
