import torch
import torch.nn as nn
from transformers import AutoModel, AutoProcessor, CLIPModel

class VisionTextEncoder(nn.Module):
    """
    Encodes images and text instructions into a shared context representation.
    
    Defaults to using a CLIP model, but can be adapted.
    Returns a sequence of embeddings: [Image_Patches, Text_Tokens].
    """
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", freeze_backbone: bool = True):
        super().__init__()
        self.model_name = model_name
        
        # Load CLIP (Vision + Text)
        # We use the full CLIPModel to get access to both towers
        # Load model using AutoModel to support both CLIP and SigLIP
        self.backbone = AutoModel.from_pretrained(model_name)
        
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
                
        # Detect hidden sizes
        if hasattr(self.backbone.config, "vision_config"):
            self.vis_hidden = self.backbone.config.vision_config.hidden_size
            self.txt_hidden = self.backbone.config.text_config.hidden_size
        else:
            # Fallback for standard CLIP if config structure differs
            self.vis_hidden = self.backbone.vision_model.config.hidden_size
            self.txt_hidden = self.backbone.text_model.config.hidden_size
        
        # Common dimension for the TRM
        self.out_dim = 512 # configurable
        
        self.vis_proj = nn.Linear(self.vis_hidden, self.out_dim)
        self.txt_proj = nn.Linear(self.txt_hidden, self.out_dim)
        
    def forward(self, pixel_values: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor = None):
        """
        Args:
            pixel_values: (B, C, H, W)
            input_ids: (B, L)
            attention_mask: (B, L)
            
        Returns:
            context: (B, SeqLen, out_dim) where SeqLen = NumPatches + NumTokens
        """
        # 1. Vision Forward
        # get_last_hidden_state=True by default for the vision_model call inside?
        # We call vision_model directly to get patches
        inputs = {"pixel_values": pixel_values}
        vision_outputs = self.backbone.vision_model(**inputs)
        # last_hidden_state: (B, NumPatches+1, D_vis) - includes [CLS]
        img_embeds = vision_outputs.last_hidden_state 
        
        # 2. Text Forward
        text_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
        text_outputs = self.backbone.text_model(**text_inputs)
        # last_hidden_state: (B, L, D_txt)
        txt_embeds = text_outputs.last_hidden_state
        
        # 3. Project to shared space
        # (B, N_v, D_out)
        img_ctx = self.vis_proj(img_embeds)
        # (B, N_t, D_out)
        txt_ctx = self.txt_proj(txt_embeds)
        
        # 4. Concatenate
        # context: (B, N_v + N_t, D_out)
        context = torch.cat([img_ctx, txt_ctx], dim=1)
        
        return context

if __name__ == "__main__":
    # Test
    print("Initializing encoder...")
    enc = VisionTextEncoder()
    print("Encoder initialized.")
    
    # Dummy data
    B = 2
    img = torch.randn(B, 3, 224, 224)
    # CLIP vocab size is ~49408
    txt = torch.randint(0, 1000, (B, 10))
    
    out = enc(img, txt)
    print("Output shape:", out.shape)
