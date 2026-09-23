from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.dataset import CrowdHumanDataset
from src.detr_data import (
    build_processor,
    make_coco_target,
    make_collate_fn,
)


# ------------------------------------------------------------
# 1. Paths
# ------------------------------------------------------------

PROJECT_ROOT = Path.cwd()

MANIFEST = (
    PROJECT_ROOT
    / "data"
    / "manifests"
    / "crowdhuman_train.jsonl"
)

print("Project root:")
print(PROJECT_ROOT)

print("\nManifest:")
print(MANIFEST)

assert MANIFEST.exists(), (
    f"\nManifest does not exist:\n{MANIFEST}\n"
    "Run this script from the robotsafe-detr project root."
)


# ------------------------------------------------------------
# 2. Load our Dataset
# ------------------------------------------------------------

dataset = CrowdHumanDataset(
    MANIFEST
)

print("\nDataset size:")
print(len(dataset))


# ------------------------------------------------------------
# 3. Find a sample with at least one usable annotation
# ------------------------------------------------------------

sample = None

for i in range(len(dataset)):

    candidate = dataset[i]

    if len(candidate["annotations"]) > 0:
        sample = candidate
        break

assert sample is not None


print("\nSample")
print("-----------------------------")
print("image_id:", sample["image_id"])
print("CrowdHuman ID:", sample["crowdhuman_id"])
print("image size:", sample["image"].size)
print("persons:", len(sample["annotations"]))
print(
    "ignored:",
    len(sample["ignored_annotations"])
)


# ------------------------------------------------------------
# 4. Inspect raw CrowdHuman box
# ------------------------------------------------------------

raw_box = sample["annotations"][0]["bbox_xywh"]

print("\nOriginal CrowdHuman fbox:")
print(raw_box)

print(
    "format: [x, y, width, height], pixels"
)


# ------------------------------------------------------------
# 5. Convert to COCO target
# ------------------------------------------------------------

target = make_coco_target(sample)

print("\nCOCO target keys:")
print(target.keys())

print("\nFirst COCO annotation:")
print(target["annotations"][0])


# ------------------------------------------------------------
# 6. Load Deformable-DETR image processor
# ------------------------------------------------------------

processor = build_processor()

print("\nProcessor:")
print(type(processor))

print("\nProcessor resize configuration:")
print(processor.size)


# ------------------------------------------------------------
# 7. Process ONE image
# ------------------------------------------------------------

encoding = processor(
    images=sample["image"],
    annotations=target,
    return_tensors="pt",
)

print("\nEncoding keys:")
print(encoding.keys())


# ------------------------------------------------------------
# 8. Inspect image tensor
# ------------------------------------------------------------

pixel_values = encoding["pixel_values"]

print("\npixel_values")
print("-----------------------------")
print("shape:", pixel_values.shape)
print("dtype:", pixel_values.dtype)

print(
    "min/max:",
    pixel_values.min().item(),
    pixel_values.max().item(),
)


# ------------------------------------------------------------
# 9. Inspect pixel mask
# ------------------------------------------------------------

pixel_mask = encoding["pixel_mask"]

print("\npixel_mask")
print("-----------------------------")
print("shape:", pixel_mask.shape)
print("dtype:", pixel_mask.dtype)

print(
    "unique values:",
    torch.unique(pixel_mask)
)


# ------------------------------------------------------------
# 10. Inspect DETR labels
# ------------------------------------------------------------

labels = encoding["labels"][0]

print("\nLabel dictionary keys:")
print(labels.keys())

print("\nNumber of objects:")
print(labels["boxes"].shape[0])

print("\nFirst 5 transformed boxes:")
print(labels["boxes"][:5])

print("\nFirst 5 class labels:")
print(labels["class_labels"][:5])


# ------------------------------------------------------------
# 11. Manually verify first box
# ------------------------------------------------------------

W, H = sample["image"].size

x, y, w, h = raw_box

expected_box = torch.tensor(
    [
        (x + w / 2) / W,
        (y + h / 2) / H,
        w / W,
        h / H,
    ],
    dtype=torch.float32,
)

actual_box = labels["boxes"][0].cpu()

print("\nManual box verification")
print("-----------------------------")

print("Expected normalized:")
print(expected_box)

print("\nProcessor produced:")
print(actual_box)

print("\nAbsolute difference:")
print(torch.abs(expected_box - actual_box))


# ------------------------------------------------------------
# 12. Build DataLoader
# ------------------------------------------------------------

collate_fn = make_collate_fn(
    processor
)

loader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=False,

    # Keep zero until pipeline is proven correct.
    num_workers=0,

    collate_fn=collate_fn,
)


# ------------------------------------------------------------
# 13. Load one batch
# ------------------------------------------------------------

batch = next(iter(loader))

print("\n\nBATCH TEST")
print("=============================")

print("\npixel_values:")
print(batch["pixel_values"].shape)

print("\npixel_mask:")
print(batch["pixel_mask"].shape)

print("\nNumber of target dictionaries:")
print(len(batch["labels"]))


for i, target in enumerate(
    batch["labels"]
):

    print(
        f"\nImage {i}:"
    )

    print(
        "objects:",
        target["boxes"].shape[0]
    )

    print(
        "boxes shape:",
        target["boxes"].shape
    )

    print(
        "class_labels shape:",
        target["class_labels"].shape
    )


# ------------------------------------------------------------
# 14. Padding ratio
# ------------------------------------------------------------

for i in range(
    batch["pixel_mask"].shape[0]
):

    ratio = (
        batch["pixel_mask"][i]
        .float()
        .mean()
        .item()
    )

    print(
        f"\nImage {i} real-pixel ratio:"
        f" {ratio:.4f}"
    )


print("\nStage 3 passed.")