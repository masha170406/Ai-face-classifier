import os
import random

from PIL import Image

from torch.utils.data import Dataset
from torchvision import transforms

# TRANSFORMS
train_transform = transforms.Compose([

    transforms.Resize((256, 256)),

    transforms.RandomResizedCrop(
        224,
        scale=(0.85, 1.0)
    ),

    transforms.RandomHorizontalFlip(p=0.5),

    transforms.ColorJitter(
        brightness=0.08,
        contrast=0.08,
        saturation=0.08,
        hue=0.02
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

val_transform = transforms.Compose([

    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# DATASET CLASS
class RealFakeDataset(Dataset):

    def __init__(self, samples, transform=None):

        self.samples = samples
        self.transform = transform

    def __len__(self):

        return len(self.samples)

    def __getitem__(self, idx):

        img_path, label = self.samples[idx]

        try:

            image = Image.open(img_path).convert("RGB")

        except Exception:

            # fallback image if corrupted
            image = Image.new("RGB", (224, 224))

        if self.transform:
            image = self.transform(image)

        return image, label

# HELPERS
def load_images_from_folder(folder, max_images=None):

    files = []

    for root, _, filenames in os.walk(folder):

        for f in filenames:

            if f.lower().endswith((".png", ".jpg", ".jpeg")):

                files.append(os.path.join(root, f))

    random.shuffle(files)

    if max_images:
        files = files[:max_images]

    return files


# SPLIT FUNCTION
def split_dataset(files, train_ratio=0.8):

    split_idx = int(len(files) * train_ratio)

    train_files = files[:split_idx]
    val_files = files[split_idx:]

    return train_files, val_files


# BUILD DATASETS
def build_datasets():

    train_samples = []
    val_samples = []

    # REAL IMAGES
    real_images = load_images_from_folder(
        r"dataset/train/real",
        max_images=8000
    )

    train_real, val_real = split_dataset(real_images)

    for f in train_real:
        train_samples.append((f, 0))

    for f in val_real:
        val_samples.append((f, 0))

    # ORIGINAL FAKE DATASET
    fake_original = load_images_from_folder(
        r"dataset/train/fake",
        max_images=3000
    )

    train_fake_original, val_fake_original = split_dataset(fake_original)

    for f in train_fake_original:
        train_samples.append((f, 1))

    for f in val_fake_original:
        val_samples.append((f, 1))


    # THISPERSONDOESNOTEXIST
    tpdne_fake = load_images_from_folder(
        r"dataset/tpdne_faces",
        max_images=3000
    )

    train_tpdne, val_tpdne = split_dataset(tpdne_fake)

    for f in train_tpdne:
        train_samples.append((f, 1))

    for f in val_tpdne:
        val_samples.append((f, 1))


    # STABLE DIFFUSION
    sd_fake = load_images_from_folder(
        r"dataset/stable_diffusion_faces",
        max_images=3000
    )

    train_sd, val_sd = split_dataset(sd_fake)

    for f in train_sd:
        train_samples.append((f, 1))

    for f in val_sd:
        val_samples.append((f, 1))

    # SHUFFLE
    random.shuffle(train_samples)
    random.shuffle(val_samples)

    # DATASETS
    train_dataset = RealFakeDataset(
        train_samples,
        transform=train_transform
    )

    val_dataset = RealFakeDataset(
        val_samples,
        transform=val_transform
    )

    # STATS
    print("DATASET SUMMARY")
    print(f"Train samples: {len(train_samples)}")
    print(f"Validation samples: {len(val_samples)}")

    train_real_count = sum(1 for _, l in train_samples if l == 0)
    train_fake_count = sum(1 for _, l in train_samples if l == 1)

    val_real_count = sum(1 for _, l in val_samples if l == 0)
    val_fake_count = sum(1 for _, l in val_samples if l == 1)

    print("\nTRAIN")
    print(f"Real : {train_real_count}")
    print(f"Fake : {train_fake_count}")

    print("\nVALIDATION")
    print(f"Real : {val_real_count}")
    print(f"Fake : {val_fake_count}")

    return train_dataset, val_dataset