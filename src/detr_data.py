from transformers import AutoImageProcessor


MODEL_ID = "SenseTime/deformable-detr"

PERSON_CLASS_ID = 0


def make_coco_target(sample):
    """
    Convert our CrowdHuman manifest representation into the
    COCO-detection format expected by DeformableDetrImageProcessor.

    Input annotation:
        [x, y, width, height] in pixels

    Output annotation:
        still [x, y, width, height] in pixels

    The Hugging Face image processor will later convert these to:
        [cx, cy, width, height] normalized to [0, 1]
    """

    coco_annotations = []

    for ann in sample["annotations"]:

        x, y, w, h = ann["bbox_xywh"]

        coco_annotations.append(
            {
                "image_id": sample["image_id"],
                "category_id": PERSON_CLASS_ID,
                "bbox": [x, y, w, h],
                "area": w * h,
                "iscrowd": 0,
            }
        )

    return {
        "image_id": sample["image_id"],
        "annotations": coco_annotations,
    }


def build_processor():
    """
    Load preprocessing configuration corresponding to
    SenseTime/deformable-detr.
    """

    processor = AutoImageProcessor.from_pretrained(
        MODEL_ID
    )

    return processor


def make_collate_fn(processor):
    """
    Returns a collate function for PyTorch DataLoader.

    The processor:
      - resizes images
      - normalizes pixels
      - pads images in a batch
      - creates pixel_mask
      - transforms detection annotations
    """

    def collate_fn(batch):

        images = [
            sample["image"]
            for sample in batch
        ]

        targets = [
            make_coco_target(sample)
            for sample in batch
        ]

        encoding = processor(
            images=images,
            annotations=targets,
            return_tensors="pt",
        )

        # Metadata is NOT passed into the Transformer.
        # We preserve it for evaluation/error analysis later.
        encoding["metadata"] = [
            {
                "image_id": sample["image_id"],
                "crowdhuman_id":
                    sample["crowdhuman_id"],
                "ignored_annotations":
                    sample["ignored_annotations"],
            }
            for sample in batch
        ]

        return encoding

    return collate_fn