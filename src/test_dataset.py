# %%
from dataset import CrowdHumanDataset


dataset = CrowdHumanDataset(
    "data/manifests/crowdhuman_train.jsonl"
)

print("Dataset size:", len(dataset))

sample = dataset[0]

print(sample.keys())
print("Image size:", sample["image"].size)
print("Image ID:", sample["image_id"])
print("CrowdHuman ID:", sample["crowdhuman_id"])
print("Persons:", len(sample["annotations"]))
print("Ignored:", len(sample["ignored_annotations"]))
# %%
