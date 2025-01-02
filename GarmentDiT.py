import torch
from .src.transformer_sd3_garm import SD3Transformer2DModel

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
                "pooled_projections": ("TENSOR",),  # Add pooled_projections here
            },
        }

    RETURN_TYPES = ("LATENT",)  # Outputs enhanced latent for decoding
    RETURN_NAMES = ("latent",)
    FUNCTION = "enhance_garment"
    CATEGORY = "Custom/Garment Enhancement"

    def enhance_garment(self, clip_embedding1, clip_embedding2, timestep, pooled_projections):
        """
        Enhance the latent space or latent encoding using the transformer model and CLIP embeddings.
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

        # Concatenate the two clip tensors along the last dimension
        clip_tensor = torch.cat((clip_tensor1, clip_tensor2), dim=-1)

        # Debugging: Print concatenated tensor shape
        print(f"Concatenated clip_tensor shape: {clip_tensor.shape}")

        # Calculate number of elements that need to be reshaped into [batch_size, 16, height, width]
        total_elements = clip_tensor.numel()
        required_elements = 16  # number of channels (16, from config)

        # Calculate height and width dynamically to match the total number of elements
        height_width = total_elements // required_elements  # Total elements divided by 16 channels
        height = width = int(height_width ** 0.5)  # Assuming square input tensor (height == width)

        if height * width * 16 != total_elements:
            raise ValueError("Calculated height and width do not match total number of elements.")

        # Reshape to (batch_size, 16, height, width)
        clip_tensor = clip_tensor.view(clip_tensor.shape[0], 16, height, width)

        # Pass through transformer model
        with torch.no_grad():
            output = self.transformer(
                hidden_states=clip_tensor,
                timestep=torch.tensor([timestep], dtype=torch.long),
                pooled_projections=pooled_projections,  # Pass pooled_projections as well
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
