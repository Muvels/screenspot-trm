from transformers import AutoConfig, AutoModel

model_name = "Tongyi-MiA/UI-Ins-7B"

print(f"Loading config for {model_name}...")
try:
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    print("Config loaded.")
    print(config)
    
    # Check for vision config
    if hasattr(config, "vision_config"):
        print("Found vision_config:", config.vision_config)
    else:
        print("No vision_config found in root config.")
        
except Exception as e:
    print(f"Error loading config: {e}")
