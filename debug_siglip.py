from transformers import AutoProcessor
import torch

model_name = "google/siglip-so400m-patch14-384"
print(f"Loading {model_name}...")
try:
    processor = AutoProcessor.from_pretrained(model_name)
    print("Processor loaded.")
    
    text = ["Click on the button", "Another task"]
    # Replicating the call in preprocess.py (assuming it uses these args)
    # I need to check preprocess.py to be sure of args, but let's try standard ones
    encoded = processor(text=text, padding="max_length", truncation=True, max_length=64, return_tensors="pt")
    
    print("Keys in encoded output:", encoded.keys())
    print("Input IDs shape:", encoded["input_ids"].shape)
    if "attention_mask" in encoded:
        print("Attention Mask shape:", encoded["attention_mask"].shape)
    else:
        print("NO attention_mask found!")

except Exception as e:
    print(f"Error: {e}")
