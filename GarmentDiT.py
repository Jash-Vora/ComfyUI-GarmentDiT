import torch
from torch import nn
from .src.transformer_sd3_garm import SD3Transformer2DModel


class GarmentEnhancementNode:
    """
    A custom node for garment enhancement using a pre-trained DiT model.
    """

    def __init__(self):
        super().__init__()
        # Load the transformer model
        model_path = "/kaggle/working/ComfyUI/models/DiT"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.transformer = SD3Transformer2DModel.from_pretrained(
            model_path, torch_dtype=torch.float16, local_files_only=True
        ).to(self.device)  # Move model to the device

        self.feature_projector1 = nn.Linear(1280, 2048).to(self.device, dtype=torch.float16)  # Projecting to 2048 dimensions
        self.feature_projector2 = nn.Linear(768, 2048).to(self.device, dtype=torch.float16)   # Projecting to 2048 dimensions

    @classmethod
    def INPUT_TYPES(cls) -> dict:
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

    def enhance_garment(
        self, clip_embedding1: torch.Tensor, clip_embedding2: torch.Tensor, timestep: int
    ) -> tuple:
        """
        Enhance the latent space or latent encoding using the transformer model and CLIP embeddings.
        """
        try:
            # Extract CLIP embeddings
            clip_tensor1 = clip_embedding1.last_hidden_state.to(self.device, dtype=torch.float16)
            clip_tensor2 = clip_embedding2.last_hidden_state.to(self.device, dtype=torch.float16)

            # Compute pooled_projections using image_embeds
            pooled_projections1 = clip_embedding1.image_embeds.to(self.device, dtype=torch.float16)
            pooled_projections2 = clip_embedding2.image_embeds.to(self.device, dtype=torch.float16)

            pooled_projections1 = self.feature_projector1(pooled_projections1)
            pooled_projections2 = self.feature_projector2(pooled_projections2)
            
            # Combine or average pooled projections
            pooled_projections = (pooled_projections1 + pooled_projections2) / 2

            # Debugging: Log tensor shapes
            print(f"clip_tensor1 shape: {clip_tensor1.shape}")
            print(f"clip_tensor2 shape: {clip_tensor2.shape}")
            print(f"pooled_projections shape: {pooled_projections.shape}")

            # Align dimensions if necessary
            max_dim1 = max(clip_tensor1.size(1), clip_tensor2.size(1))
            max_dim2 = max(clip_tensor1.size(2), clip_tensor2.size(2))

            if clip_tensor1.size(1) < max_dim1 or clip_tensor1.size(2) < max_dim2:
                clip_tensor1 = torch.nn.functional.pad(
                    clip_tensor1,
                    (0, max_dim2 - clip_tensor1.size(2), 0, max_dim1 - clip_tensor1.size(1)),
                )
            if clip_tensor2.size(1) < max_dim1 or clip_tensor2.size(2) < max_dim2:
                clip_tensor2 = torch.nn.functional.pad(
                    clip_tensor2,
                    (0, max_dim2 - clip_tensor2.size(2), 0, max_dim1 - clip_tensor2.size(1)),
                )

            # Concatenate tensors along the last dimension
            clip_tensor = torch.cat((clip_tensor1, clip_tensor2), dim=-1)

            # Debugging: Log concatenated tensor shape
            print(f"Concatenated clip_tensor shape: {clip_tensor.shape}")

            # Calculate height and width for reshaping
            total_elements = clip_tensor.numel()
            required_channels = 16  # Expected number of channels

            if total_elements % required_channels != 0:
                raise ValueError(
                    f"Total elements ({total_elements}) are not divisible by required channels ({required_channels})."
                )

            spatial_elements = total_elements // required_channels
            height, width = self.find_closest_factors(spatial_elements)

            if height * width != spatial_elements:
                raise ValueError(
                    f"Calculated height ({height}) and width ({width}) do not match spatial elements ({spatial_elements})."
                )

            # Reshape tensor to (batch_size, required_channels, height, width)
            clip_tensor = clip_tensor.view(
                clip_tensor.shape[0], required_channels, height, width
            )

            # Pass through transformer model
            with torch.no_grad():
                output = self.transformer(
                    hidden_states=clip_tensor,
                    timestep=torch.tensor([timestep], dtype=torch.long, device=self.device),
                    pooled_projections=pooled_projections,
                    return_dict=True,
                )

            print(output.shape)
            # Return the enhanced latent representation
            return (output.sample,)

        except Exception as e:
            raise RuntimeError(f"Error in garment enhancement: {e}")

    @staticmethod
    def find_closest_factors(n: int) -> tuple:
        """
        Finds the closest pair of factors for n and returns them as (height, width).
        """
        for i in range(int(n**0.5), 0, -1):
            if n % i == 0:
                return i, n // i
        return 1, n  # Fallback for prime numbers


# Node registration
NODE_CLASS_MAPPINGS = {
    "GarmentEnhancementNode": GarmentEnhancementNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GarmentEnhancementNode": "Garment Enhancement"
}
