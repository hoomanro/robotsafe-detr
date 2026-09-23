import json
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset


def load_jsonl(path):
    records = []

    with open(path, "r") as f:
        for line in f:
            records.append(
                json.loads(line)
            )

    return records


class CrowdHumanDataset(Dataset):

    def __init__(self, manifest_path):

        self.manifest_path = Path(
            manifest_path
        ).resolve()

        self.records = load_jsonl(
            self.manifest_path
        )

        # Expected structure:
        #
        # project/
        # ├── data/
        # │   └── manifests/
        # │       └── crowdhuman_train.jsonl
        #
        # manifest_path.parents[0] = manifests/
        # manifest_path.parents[1] = data/
        # manifest_path.parents[2] = project/
        #
        self.project_root = (
            self.manifest_path.parents[2]
        )

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):

        record = self.records[index]

        image_path = Path(
            record["image_path"]
        )

        # Manifest stores portable project-relative paths.
        if not image_path.is_absolute():
            image_path = (
                self.project_root
                / image_path
            )

        if not image_path.exists():
            raise FileNotFoundError(
                f"\nImage not found:\n"
                f"{image_path}\n\n"
                f"Manifest entry:\n"
                f"{record['image_path']}\n\n"
                f"Project root:\n"
                f"{self.project_root}"
            )

        image = Image.open(
            image_path
        ).convert("RGB")

        return {
            "image": image,

            "image_id":
                record["image_id"],

            "crowdhuman_id":
                record["crowdhuman_id"],

            "annotations":
                record["annotations"],

            "ignored_annotations":
                record[
                    "ignored_annotations"
                ],

            # Useful later for debugging/evaluation
            "image_path":
                str(image_path),
        }