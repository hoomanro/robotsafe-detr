import json
import numpy as np
from collections import Counter


def load_jsonl(path):

    with open(path) as f:
        return [
            json.loads(line)
            for line in f
        ]


records = load_jsonl(
    "data/manifests/crowdhuman_train.jsonl"
)

persons_per_image = [
    len(r["annotations"])
    for r in records
]

print("Images:", len(records))

print(
    "Mean persons/image:",
    np.mean(persons_per_image)
)

print(
    "Median:",
    np.median(persons_per_image)
)

print(
    "Maximum:",
    np.max(persons_per_image)
)


occlusion_counts = Counter()

for record in records:

    for ann in record["annotations"]:

        occlusion_counts[
            ann["occlusion"]
        ] += 1


print(
    "Occlusion:",
    occlusion_counts
)