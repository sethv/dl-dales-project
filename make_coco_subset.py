# Take in a COCO annotations JSON and reduce it to a subset of images/annotations
# Run this from project root directory
# Then download images to train2017 folder with
# xargs -n 1 curl -O < urls.txt
import json
import collections


# Set up args here
old_anns_name = "instances_val2017"
new_anns_name = "instances_val2017_5classes_areagt100"

input_json_file = f"Deformable-DETR/data/coco/annotations/{old_anns_name}.json"
output_json_file = f"Deformable-DETR/data/coco/annotations/{new_anns_name}.json"
url_file = f"urls_{new_anns_name}.txt"

category_ids_to_keep = {
    # 1, # person
    5, # airplane
    7, # train
    # 13, # stop sign
    # 16, # bird
    # 17, # cat
    # 22, # elephant
    23, # bear
    24, # zebra
    25, # giraffe
    # 38, # kite
    # 44, # bottle
    # 77, # cell phone
    # 82, # refrigerator
    # 88, # teddy bear (see if it can tell apart from bear)
}
max_images = 1000000000  # how many images to keep
keep_iscrowd = True  # ignore annotations with 'iscrowd' = 1
min_area = 100  # remove any annotations with 'area' < this (in pixels)


# Go through annotations file

print("Loading original annotations file: ", input_json_file)
with open(input_json_file) as f:
    anns = json.load(f)

print("Number of images: ", len(anns["images"]))
print("Number of annotations: ", len(anns["annotations"]))
print("Number of categories: ", len(anns["categories"]))
print("\t", ", ".join(
    f"{c['id']}={c['name']}"
    for c in anns["categories"]
))

# import numpy as np
# areas = [ann["area"] for ann in anns]
# avg_area = np.median(areas)
# print("average area in original annotations", avg_area)

# This new dict will be filled with only some images
# We only keep the annotations + classes that match these images
output_anns = {k: anns[k] for k in ["info", "licenses"]}
# TODO save the args used to generate this new dataset
output_anns["images"] = []
output_anns["annotations"] = []
output_anns["categories"] = []

temp_anns = collections.defaultdict(list) # map images->annotations
image_ids_to_keep = set()
urls = []  # image URLs

for ann in anns["annotations"]:
    # input("hit enter for next:")
    # print(ann)
    # if ann["image_id"] in image_ids_to_keep:
    if ann["category_id"] not in category_ids_to_keep:
        # print("\twrong class")
        continue
    if not keep_iscrowd and ann["iscrowd"] == 1:
        # print("\tiscrowd")
        continue
    if ann["area"] < min_area:
        # print("\tarea too small")
        continue
    else:
        # category_ids_to_keep.add(ann["category_id"])
        # print("\tkeep, image_id=", ann["image_id"])
        # output_anns["annotations"].append(ann)
        ann.pop("segmentation", 'alreadygone')  # getting in the way
        temp_anns[ann["image_id"]].append(ann)
        image_ids_to_keep.add(ann["image_id"])

num_images_used = 0
for image in anns["images"]:
    if image["id"] in image_ids_to_keep:
        num_images_used += 1
        # print(image["coco_url"])
        urls.append(image["coco_url"])
        output_anns["images"].append(image)
        output_anns["annotations"].extend(temp_anns[image["id"]])
    if num_images_used == max_images:
        # Stop, we have enough images already
        break

    



for cat in anns["categories"]:
    if cat["id"] in category_ids_to_keep:
        output_anns["categories"].append(cat)

print("New annotations file: ", output_json_file)
print("Number of images: ", len(output_anns["images"]))
print("Number of annotations: ", len(output_anns["annotations"]))
print("Number of categories: ", len(output_anns["categories"]))
print("\t", ", ".join(
    f"{c['id']}={c['name']}"
    for c in output_anns["categories"]
))

with open(output_json_file, "w") as of:
    # Save the new reduced annotations JSON
    json.dump(output_anns, of, indent=2)

# Save URLs to txt file so we can download some images easily
with open(url_file, "w") as uf:
    uf.writelines([f"{url}\n" for url in urls])

