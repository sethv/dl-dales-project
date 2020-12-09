# Take in a COCO annotations JSON and reduce it to a subset of images/annotations
# Run this from project root directory
# Then download images to train2017 folder with
# xargs -n 1 curl -O < urls.txt
import json

input_json_file = "Deformable-DETR/data/coco/annotations/instances_train2017_orig.json"
output_json_file = "Deformable-DETR/data/coco/annotations/instances_train2017.json"
url_file = "urls.txt"
num_images = 20  # how many images
category_ids_to_keep = {1}  # keep only person instances
filtering_classes = len(category_ids_to_keep) > 0  # True/False

print("Loading original annotations file: ", input_json_file)
with open(input_json_file) as f:
    anns = json.load(f)

print("Number of images: ", len(anns["images"]))
print("Number of annotations: ", len(anns["annotations"]))
print("Number of categories: ", len(anns["categories"]))

# This new dict will be filled with only some images
# We only keep the annotations + classes that match these images
output_anns = {k: anns[k] for k in ["info", "licenses"]}
output_anns["images"] = []
output_anns["annotations"] = []
output_anns["categories"] = []

image_ids_to_keep = set()
urls = []  # image URLs

for image in anns["images"][:num_images]:
    print(image["coco_url"])
    urls.append(image["coco_url"])
    image_ids_to_keep.add(image["id"])
    output_anns["images"].append(image)


for ann in anns["annotations"]:
    if ann["image_id"] in image_ids_to_keep:
        if filtering_classes and ann["category_id"] not in category_ids_to_keep:
            pass
        else:
            category_ids_to_keep.add(ann["category_id"])
            output_anns["annotations"].append(ann)

for cat in anns["categories"]:
    if cat["id"] in category_ids_to_keep:
        output_anns["categories"].append(cat)

print("New annotations file: ", output_json_file)
print("Number of images: ", len(output_anns["images"]))
print("Number of annotations: ", len(output_anns["annotations"]))
print("Number of categories: ", len(output_anns["categories"]))
with open(output_json_file, "w") as of:
    # Save the new reduced annotations JSON
    json.dump(output_anns, of, indent=2)

# Save URLs to txt file so we can download some images easily
with open(url_file, "w") as uf:
    uf.writelines([f"{url}\n" for url in urls])
