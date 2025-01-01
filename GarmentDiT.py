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

    def enhance_garment(self, latent_image, clip_embeddings):
        """
        Enhance the input latent image using the transformer model and CLIP embeddings.
        """
        latent_image_tensor = latent_image["samples"]  # Extract latent tensor
        clip_tensor = clip_embeddings["embedding"]  # Extract CLIP embeddings

        # Run the transformer model
        with torch.no_grad():
            enhanced_latent = self.transformer(latent_image_tensor, encoder_hidden_states=clip_tensor).sample

        # Return enhanced latent in the expected format
        return ({
            "samples": enhanced_latent,
            "shape": latent_image["shape"]
        },)


# Node registration
NODE_CLASS_MAPPINGS = {
    "GarmentEnhancementNode": GarmentEnhancementNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GarmentEnhancementNode": "Garment Enhancement"
}
