import json

path = "data/crowdhuman/annotation_train.odgt"

with open(path, "r") as f:
    line = f.readline()

record = json.loads(line)

print("Image ID:", record["ID"])
print("Number of annotations:", len(record["gtboxes"]))

for gt in record["gtboxes"][:5]:
    print()
    print("tag:", gt["tag"])
    print("fbox:", gt["fbox"])
    print("vbox:", gt["vbox"])
    print("hbox:", gt["hbox"])
    print("extra:", gt.get("extra", {}))