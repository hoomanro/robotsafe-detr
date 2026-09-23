import json
import random
from pathlib import Path

from PIL import Image, ImageDraw


MANIFEST = "data/manifests/crowdhuman_train.jsonl"
OUTPUT_DIR = Path("outputs/qc")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


with open(MANIFEST) as f:
    records = [
        json.loads(line)
        for line in f
    ]


record = random.choice(records)

print("Image:", record["image_path"])
print("CrowdHuman ID:", record["crowdhuman_id"])
print("People:", len(record["annotations"]))


def draw_boxes(record, box_key, output_name):

    image = Image.open(
        record["image_path"]
    ).convert("RGB")

    draw = ImageDraw.Draw(image)

    count = 0

    for ann in record["annotations"]:

        box = ann.get(box_key)

        if box is None:
            continue

        x, y, w, h = box

        draw.rectangle(
            [
                x,
                y,
                x + w,
                y + h,
            ],
            width=3,
        )

        count += 1

    output_path = OUTPUT_DIR / output_name

    image.save(output_path)

    print(
        f"{box_key}: drew {count} boxes"
        f" -> {output_path}"
    )


draw_boxes(
    record,
    "bbox_xywh",
    "full_body.png"
)

draw_boxes(
    record,
    "visible_bbox_xywh",
    "visible_body.png"
)

draw_boxes(
    record,
    "head_bbox_xywh",
    "head.png"
)