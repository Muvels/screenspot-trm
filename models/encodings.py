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
        
        # Load Model (Handle Qwen2.5-VL vs others)
        if "UI-Ins-7B" in model_name or "Qwen" in model_name:
            # Fallback imports if needed, but AutoModel is usually fine if trust_remote_code=True
            self.is_qwen = True
            try:
                # Load the full model to extract vision tower
                # We use AutoModelForCausalLM or similar usually, but let's try generic AutoModel first
                # Qwen-VL often needs specific class for full functionality, but for just vision tower:
                self.backbone = AutoModel.from_pretrained(model_name, trust_remote_code=True)
                
                # Extract Vision Tower
                # Qwen2.5-VL structure: model.visual
                if hasattr(self.backbone, "visual"):
                    self.vision_tower = self.backbone.visual
                elif hasattr(self.backbone, "model") and hasattr(self.backbone.model, "visual"):
                    self.vision_tower = self.backbone.model.visual
                else:
                    raise ValueError("Could not find 'visual' module in Qwen model.")
                    
                # Config params
                self.vis_hidden = self.vision_tower.config.hidden_size
                # Text tower ?? For now assume we might still use CLIPlike text or just ignore text if Qwen handles both?
                # The user request said "extract... vision tower". Using Qwen's text tower is complex (LLM).
                # Current Agent structure uses separate text encoding usually?
                # Wait, existing code uses `CLIPModel` which has BOTH.
                # If we switch to Qwen Vision, what do we use for Text? 
                # The user prompt implies replacing the vision part. 
                # However, Qwen-VL is a VLM. Embedding text with it is essentially running the LLM?
                # For `VisionTextEncoder` which returns SEPARATE embeddings:
                # Retaining CLIP for text might be safer if we only want vision from Qwen.
                # BUT, let's assume for now we might leave text empty or use a small text encoder?
                # Actually, `InfoMaxAgent` expects `txt_ctx` from `encoder`.
                # FIX: Let's keep a small CLIP for text if we use Qwen for Vision, OR
                # if the user wants purely Qwen, we'd need to run the LLM part for text embeddings.
                # Running 7B LLM just for text embedding in this small "TinyRecursiveModel" setup seems overkill/slow?
                # Let's assume we keep CLIP for text for now to match the "extract vision tower" intent,
                # or simpler: Just fail on text for now/dummy it if we only care about vision?
                # Let's try to keep CLIP for text fallback if Qwen is used for Vision.
                
                self.text_backbone = CLIPModel.from_pretrained("openai/clip-vit-base-patch32") # Fallback text
                self.txt_hidden = self.text_backbone.text_model.config.hidden_size
                
            except Exception as e:
                print(f"Error loading Qwen model: {e}")
                raise e
        else:
            self.is_qwen = False
            # Standard CLIP
            self.backbone = AutoModel.from_pretrained(model_name)
            self.vision_tower = self.backbone.vision_model
            self.text_backbone = self.backbone if hasattr(self.backbone, "text_model") else None # CLIP has it inside
            
            if hasattr(self.backbone.config, "vision_config"):
                self.vis_hidden = self.backbone.config.vision_config.hidden_size
                self.txt_hidden = self.backbone.config.text_config.hidden_size
            else:
                self.vis_hidden = self.backbone.vision_model.config.hidden_size
                self.txt_hidden = self.backbone.text_model.config.hidden_size
        
        if freeze_backbone:
            # Freeze Qwen Vision
            if self.is_qwen:
                for param in self.vision_tower.parameters():
                    param.requires_grad = False
                for param in self.text_backbone.parameters():
                    param.requires_grad = False
            else:
                for param in self.backbone.parameters():
                    param.requires_grad = False
                
        # Common dimension for the TRM
        self.out_dim = 512 # configurable
        
        self.vis_proj = nn.Linear(self.vis_hidden, self.out_dim)
        self.txt_proj = nn.Linear(self.txt_hidden, self.out_dim)
        
    def forward(self, pixel_values: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor = None, **kwargs):
        """
        Args:
            pixel_values: (B, C, H, W) OR (Order dictionary/flat tensor for Qwen)
            input_ids: (B, L)
            attention_mask: (B, L)
            kwargs: extra args like 'grid_thw' for Qwen
            
        Returns:
            context: (B, SeqLen, out_dim) where SeqLen = NumPatches + NumTokens
        """
        # 1. Vision Forward
        if self.is_qwen:
            # Qwen Vision Forward
            # Expecting pixel_values to be a tensor of all flattened patches? 
            # Or formatted by processor.
            # Qwen visual forward usually: (hidden_states, grid_thw)
            # Check if grid_thw is passed
            grid_thw = kwargs.get("image_grid_thw", None)
            
            # If pixel_values is standard 4D tensor, Qwen might complain if it expects flattened?
            # Qwen2.5-VL processor outputs 'pixel_values' (N, D) and 'image_grid_thw'
            
            vision_outputs = self.vision_tower(pixel_values, grid_thw=grid_thw)
            
            # Output is usually just the hidden states (N_total, D) 
            # It is NOT batched (B, ...) if generated by processor with padding removed?
            # Actually Qwen-VL visual output is often (Sum(Tokens), D).
            # We need to reshape/split back to batch if possible, or TRM needs to handle variable len?
            # TRM expects (B, Seq, H).
            # This is tricky with Qwen-VL's efficiency packing (var length).
            # For simplicity, if batch size > 1, this needs careful handling or padding.
            # Let's assume for now we might have to pad manually or use batch size 1 for safety?
            # Or reconstruct batch from grid_thw?
            # grid_thw is (B, 3) usually? No, it's (NumImages, 3).
            
            img_embeds = vision_outputs # (TotalTokens, D)
            
            # We need to restructure to (B, MaxTokens, D) to work with current TRM
            # simple reshape if all images same size?
            # If standard dataloader resizes to fixed:
            # But Qwen processor usually preserves ratio -> var size.
            
            # FIX: If we can't easily batch, maybe we just treat it as flattened batch?
            # But we need to concat with text (B, L, D).
            # Let's attempt to split by grid_thw if present.
            if grid_thw is not None:
                # Calculate tokens per image
                # grid_thw shape (N_images, 3) -> T*H*W
                split_sizes = [g[0]*g[1]*g[2] for g in grid_thw]
                img_embeds_list = torch.split(img_embeds, split_sizes)
                # Pad to max
                img_embeds = torch.nn.utils.rnn.pad_sequence(img_embeds_list, batch_first=True)
            else:
                # Fallback if standard 3D/4D input (unlikely for Qwen Visual)
                pass

        else:
            inputs = {"pixel_values": pixel_values}
            vision_outputs = self.backbone.vision_model(**inputs)
            # last_hidden_state: (B, NumPatches+1, D_vis) - includes [CLS]
            img_embeds = vision_outputs.last_hidden_state 
        
        # 2. Text Forward
        if self.is_qwen:
            # Use Fallback CLIP for Text
            text_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
            text_outputs = self.text_backbone.text_model(**text_inputs)
            txt_embeds = text_outputs.last_hidden_state
        else:
            text_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
            text_outputs = self.backbone.text_model(**text_inputs)
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
