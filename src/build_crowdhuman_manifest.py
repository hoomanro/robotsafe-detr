import json
from pathlib import Path


ROOT = Path("data/crowdhuman")

TRAIN_ANNOTATIONS = ROOT / "annotation_train.odgt"
VAL_ANNOTATIONS = ROOT / "annotation_val.odgt"

# Change this after looking at your extracted directory.
IMAGE_ROOT = ROOT / "Images"

OUTPUT_DIR = Path("data/manifests")


def load_odgt(path):
    records = []

    with open(path, "r") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            records.append(json.loads(line))

    return records


def convert_record(record, image_id):

    image_name = f"{record['ID']}.jpg"
    image_path = IMAGE_ROOT / image_name

    annotations = []
    ignored_annotations = []

    for gt in record["gtboxes"]:

        tag = gt.get("tag", "")

        if tag != "person":
            continue

        x, y, w, h = map(float, gt["fbox"])

        # Invalid geometry
        if w <= 0 or h <= 0:
            continue

        extra = gt.get("extra", {})

        annotation = {
            "bbox_xywh": [x, y, w, h],
            "visible_bbox_xywh": gt.get("vbox"),
            "head_bbox_xywh": gt.get("hbox"),
            "occlusion": extra.get("occ"),
            "box_id": extra.get("box_id"),
        }

        if extra.get("ignore", 0) == 1:
            ignored_annotations.append(annotation)
        else:
            annotations.append(annotation)

    return {
        "image_id": image_id,
        "crowdhuman_id": record["ID"],
        "image_path": str(image_path),
        "annotations": annotations,
        "ignored_annotations": ignored_annotations,
    }


def build_manifest(annotation_path):

    raw_records = load_odgt(annotation_path)

    output = []

    for image_id, record in enumerate(raw_records):

        converted = convert_record(
            record,
            image_id=image_id,
        )

        if not Path(converted["image_path"]).exists():
            print(
                "WARNING: missing image:",
                converted["image_path"]
            )
            continue

        output.append(converted)

    return output


def write_jsonl(records, path):

    with open(path, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    train = build_manifest(
        TRAIN_ANNOTATIONS
    )

    val = build_manifest(
        VAL_ANNOTATIONS
    )

    write_jsonl(
        train,
        OUTPUT_DIR / "crowdhuman_train.jsonl"
    )

    write_jsonl(
        val,
        OUTPUT_DIR / "crowdhuman_val.jsonl"
    )

    print("Training images:", len(train))
    print("Validation images:", len(val))


if __name__ == "__main__":
    main()