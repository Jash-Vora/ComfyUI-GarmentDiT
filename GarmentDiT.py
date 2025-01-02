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

        # Concatenate the tensors
        clip_tensor = torch.cat((clip_tensor1, clip_tensor2), dim=1)

        # Add timestep information (expand and concatenate if needed)
        timestep_tensor = torch.tensor([timestep], dtype=torch.float16).to(clip_tensor.device)
        timestep_tensor = timestep_tensor.unsqueeze(0).expand(clip_tensor.size(0), -1)  # Expand to batch size
        clip_tensor = torch.cat((clip_tensor, timestep_tensor), dim=1)

        # The transformer should directly use CLIP embeddings for enhancement
        with torch.no_grad():
            enhanced_latent = self.transformer(clip_tensor).sample  # Use clip_tensor directly

        # Return the enhanced latent representation (without needing to manually generate latent_image_tensor)
        return ({
            "samples": enhanced_latent,
            "shape": enhanced_latent.shape  # Shape of the enhanced latent
        },)

# Node registration
NODE_CLASS_MAPPINGS = {
    "GarmentEnhancementNode": GarmentEnhancementNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GarmentEnhancementNode": "Garment Enhancement"
}
