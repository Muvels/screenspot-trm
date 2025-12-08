"""
Script to merge two ScreenSpot datasets into a unified format.

Datasets:
- mlfoundations-cua-dev/easyr1-screenspot-pro-eval (small, needs processing)
- HyperCluster/OS-Atlas_ScreenSpot (large, used as base)

Output columns: image, task, image_width, image_height, bbox
"""

from datasets import load_dataset, concatenate_datasets, Features, Image, Value, Sequence
from pathlib import Path


def process_easyr1_screenspot(ds, target_features):
    """
    Process mlfoundations-cua-dev/easyr1-screenspot-pro-eval dataset.
    
    Column mappings:
    - images (list) -> image (take first item)
    - easyr1_prompt -> task
    - image_width -> image_width
    - image_height -> image_height
    - normalized_bbox -> bbox
    """
    def transform(example):
        # images is a list, take the first one
        images = example["images"]
        image = images[0] if isinstance(images, list) and len(images) > 0 else images
        
        # Convert bbox to list of float (match target type)
        bbox = example["normalized_bbox"]
        bbox = [float(x) for x in bbox] if bbox else []
        
        return {
            "image": image,
            "task": example["easyr1_prompt"],
            "image_width": int(example["image_width"]),
            "image_height": int(example["image_height"]),
            "bbox": bbox,
        }
    
    processed = ds.map(transform, remove_columns=ds.column_names)
    return processed.cast(target_features)


def process_os_atlas_screenspot(ds):
    """
    Process HyperCluster/OS-Atlas_ScreenSpot dataset.
    
    Just rename columns - no row-by-row transformation needed.
    - image -> image (already correct)
    - task -> task (already correct)
    - img_size -> split into image_width and image_height
    - bbox -> bbox (already correct)
    """
    # Rename img_size columns using a simple select
    def transform(example):
        img_size = example["img_size"]
        width, height = int(img_size[0]), int(img_size[1])
        return {
            "image_width": width,
            "image_height": height,
        }
    
    # Only add the new columns, keep existing ones
    ds = ds.map(transform)
    # Remove img_size and select final columns in order
    ds = ds.remove_columns(["img_size"])
    # Reorder columns
    ds = ds.select_columns(["image", "task", "image_width", "image_height", "bbox"])
    return ds


def main():
    output_dir = Path("dataset")
    output_dir.mkdir(exist_ok=True)
    
    # Load the large dataset first (as base)
    print("Loading HyperCluster/OS-Atlas_ScreenSpot (large, base dataset)...")
    ds_atlas = load_dataset("HyperCluster/OS-Atlas_ScreenSpot")
    
    # Process the large dataset with minimal transformation
    processed_atlas = []
    for split_name in ds_atlas.keys():
        print(f"Processing OS-Atlas_ScreenSpot/{split_name}...")
        print(f"  Columns: {ds_atlas[split_name].column_names}")
        print(f"  Examples: {len(ds_atlas[split_name])}")
        processed = process_os_atlas_screenspot(ds_atlas[split_name])
        processed_atlas.append(processed)
    
    # Combine atlas splits if multiple
    base_dataset = concatenate_datasets(processed_atlas) if len(processed_atlas) > 1 else processed_atlas[0]
    print(f"Base dataset ready: {len(base_dataset)} examples")
    print(f"  Features: {base_dataset.features}")
    
    # Now load and process the smaller dataset
    print("\nLoading mlfoundations-cua-dev/easyr1-screenspot-pro-eval (small dataset)...")
    ds_easyr1 = load_dataset("mlfoundations-cua-dev/easyr1-screenspot-pro-eval")
    
    processed_easyr1 = []
    for split_name in ds_easyr1.keys():
        print(f"Processing easyr1-screenspot-pro-eval/{split_name}...")
        print(f"  Columns: {ds_easyr1[split_name].column_names}")
        print(f"  Examples: {len(ds_easyr1[split_name])}")
        # Cast to match the base dataset's features
        processed = process_easyr1_screenspot(ds_easyr1[split_name], base_dataset.features)
        processed_easyr1.append(processed)
    
    # Merge all datasets
    print("\nMerging datasets...")
    all_datasets = [base_dataset] + processed_easyr1
    merged_dataset = concatenate_datasets(all_datasets)
    print(f"Total merged examples: {len(merged_dataset)}")
    
    # Save as parquet
    output_path = output_dir / "merged_screenspot.parquet"
    print(f"\nSaving to {output_path}...")
    merged_dataset.to_parquet(str(output_path))
    
    print(f"\nDone! Merged dataset saved to {output_path}")
    print(f"Columns: {merged_dataset.column_names}")
    print(f"Total rows: {len(merged_dataset)}")


if __name__ == "__main__":
    main()
