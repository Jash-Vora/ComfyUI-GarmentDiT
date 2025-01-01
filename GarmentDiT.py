import torch
from safetensors.torch import load_file
from comfy.model_base import CustomNode, comfy_model
from .src.transformer_sd3_garm import SD3Transformer2DModel  # Replace with the correct import for your model class

class TransformerEnhancementNode(CustomNode):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent_input": ("LATENT",),  # Input from the VAE encoder
                "clip_context": ("CLIP",),   # Input from the CLIP node
                "timestep": ("INT",),        # Optional: timestep for the transformer
            }
        }

    @classmethod
    def OUTPUT_TYPES(cls):
        return {
            "enhanced_latent": ("LATENT",)
        }

    def __init__(self):
        super().__init__()
        # Define the path to the .safetensors file
        transformer_model_path = "/kaggle/working/ComfyUI/models/DiT/diffusion_pytorch_model.safetensors"
        print(f"Loading transformer model from: {transformer_model_path}")
        
        # Load the model weights
        state_dict = load_file(transformer_model_path)  # Load safetensors state dict
        self.transformer = SD3Transformer2DModel()  # Replace with your actual model class
        self.transformer.load_state_dict(state_dict)  # Load weights
        self.transformer.eval()  # Set the model to evaluation mode

    def forward(self, latent_input, clip_context, timestep=None):
        # Ensure timestep is a tensor
        if timestep is None:
            timestep = torch.tensor(1, dtype=torch.long)  # Default timestep if not provided

        # Forward pass through the transformer
        with torch.no_grad():
            enhanced_latent = self.transformer(
                hidden_states=latent_input,
                encoder_hidden_states=clip_context,
                timestep=timestep,
            ).sample

        return {"enhanced_latent": enhanced_latent}


# Register the node in ComfyUI
comfy_model.register_node(
    "TransformerEnhancementNode",
    TransformerEnhancementNode,
    inputs=TransformerEnhancementNode.INPUT_TYPES(),
    outputs=TransformerEnhancementNode.OUTPUT_TYPES(),
)
