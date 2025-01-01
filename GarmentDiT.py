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
        model_path = "/kaggle/working/ComfyUI/models/DiT/"
        self.transformer = SD3Transformer2DModel.from_pretrained(
            model_path, torch_dtype=torch.float16, local_files_only=True
        )
        self.transformer.eval()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip_embeddings": ("CLIP_VISION_OUTPUT",),  # CLIP embeddings
            },
        }

    RETURN_TYPES = ("LATENT",)  # Outputs enhanced latent for decoding
    FUNCTION = "enhance_garment"
    CATEGORY = "Custom/Garment Enhancement"

    def enhance_garment(self, clip_embeddings):
        """
        Enhance the latent space or latent encoding using the transformer model and CLIP embeddings.
        No latent_image_tensor is used since the CLIP embeddings directly guide the enhancement.
        """
        print(dir(clip_embeddings))  # To inspect the structure
        # Extract CLIP embeddings (this contains the visual feature information)
        clip_tensor = clip_embeddings["embedding"]

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
