import json
import random

from PIL import Image, ImageDraw


with open(
    "data/manifests/crowdhuman_train.jsonl"
) as f:
    records = [
        json.loads(line)
        for line in f
    ]


record = random.choice(records)

image = Image.open(
    record["image_path"]
).convert("RGB")

draw = ImageDraw.Draw(image)


for ann in record["annotations"]:

    x, y, w, h = ann["bbox_xywh"]

    draw.rectangle(
        [
            x,
            y,
            x + w,
            y + h,
        ],
        width=3,
    )


from pathlib import Path

Path("outputs/qc").mkdir(parents=True, exist_ok=True)

output_path = "outputs/qc/sample_boxes.png"
image.save(output_path)

print(f"Saved to {output_path}")