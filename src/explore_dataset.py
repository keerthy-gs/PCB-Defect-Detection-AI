from pathlib import Path

# Path to the dataset
dataset_path = Path("dataset/DeepPCB")

splits = ["train", "valid", "test"]

print("=" * 50)
print("PCB DATASET SUMMARY")
print("=" * 50)

for split in splits:
    image_folder = dataset_path / split / "images"
    label_folder = dataset_path / split / "labels"

    image_count = len(list(image_folder.glob("*")))
    label_count = len(list(label_folder.glob("*.txt")))

    print(f"\n{split.upper()}")
    print(f"Images : {image_count}")
    print(f"Labels : {label_count}")

print("\nDone!")