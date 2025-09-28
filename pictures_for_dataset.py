# ✅ Скрипт для формування повного датасету з COCO під 5 класів (з онлайн-завантаженням зображень)

import os
import random
import shutil
import requests
from tqdm import tqdm
from collections import defaultdict
from pycocotools.coco import COCO

# === Конфігурація ===
ANNOTATION_FILE = 'annotations/instances_train2017.json'  # Попередньо розпакуй
COCO_IMAGES_URL = 'http://images.cocodataset.org/train2017/'  # URL зображень
OUTPUT_DIR = 'coco_balanced_dataset_new'
TARGET_CLASSES = ['car', 'truck', 'bus', 'motorcycle', 'person']
TARGET_OBJECTS_PER_CLASS = 500  # мінімум 200, бажано 500
SPLIT_RATIO = {'train': 0.7, 'val': 0.2, 'test': 0.1}

# === Підготовка ===
os.makedirs(OUTPUT_DIR, exist_ok=True)
for split in ['train', 'val', 'test']:
    os.makedirs(os.path.join(OUTPUT_DIR, 'images', split), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, 'labels', split), exist_ok=True)

# === Завантаження COCO ===
coco = COCO(ANNOTATION_FILE)
class_name_to_id = {cls: coco.getCatIds(catNms=[cls])[0] for cls in TARGET_CLASSES}
class_id_to_index = {v: i for i, v in enumerate(class_name_to_id.values())}

# === Вибірка унікальних зображень
img_to_anns = defaultdict(list)
used_img_ids = set()
class_instance_counts = defaultdict(int)

print("\n🔍 Вибір зображень з COCO...")

for class_name in TARGET_CLASSES:
    cat_id = class_name_to_id[class_name]
    img_ids = coco.getImgIds(catIds=[cat_id])
    random.shuffle(img_ids)

    for img_id in img_ids:
        if class_instance_counts[class_name] >= TARGET_OBJECTS_PER_CLASS:
            break

        ann_ids = coco.getAnnIds(imgIds=[img_id], catIds=[cat_id], iscrowd=False)
        anns = coco.loadAnns(ann_ids)
        if not anns:
            continue

        if img_id in used_img_ids:
            continue

        used_img_ids.add(img_id)
        img_info = coco.loadImgs(img_id)[0]
        img_file = img_info['file_name']
        img_w, img_h = img_info['width'], img_info['height']

        ann_ids_all = coco.getAnnIds(imgIds=[img_id], iscrowd=False)
        anns_all = coco.loadAnns(ann_ids_all)

        yolo_lines = []
        class_ids_in_image = set()
        for ann in anns_all:
            cid = ann['category_id']
            if cid not in class_id_to_index:
                continue
            yolo_cls = class_id_to_index[cid]
            bbox = ann['bbox']
            x_c = (bbox[0] + bbox[2]/2) / img_w
            y_c = (bbox[1] + bbox[3]/2) / img_h
            w = bbox[2] / img_w
            h = bbox[3] / img_h
            yolo_lines.append(f"{yolo_cls} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}")
            class_ids_in_image.add(coco.loadCats(cid)[0]['name'])

        # оновити кількість інстансів для кожного класу
        for cls in class_ids_in_image:
            if cls in class_instance_counts:
                class_instance_counts[cls] += 1

        img_to_anns[img_file] = yolo_lines

print("\n📊 Об'єкти по класах:")
for cls in TARGET_CLASSES:
    print(f"{cls:12s}: {class_instance_counts[cls]}")

# === Розподіл на train/val/test
img_files = list(img_to_anns.keys())
random.shuffle(img_files)
n_total = len(img_files)
n_train = int(n_total * SPLIT_RATIO['train'])
n_val = int(n_total * SPLIT_RATIO['val'])
n_test = n_total - n_train - n_val
splits = {
    'train': img_files[:n_train],
    'val': img_files[n_train:n_train+n_val],
    'test': img_files[n_train+n_val:]
}

# === Збереження файлів
print("\n💾 Завантаження зображень та збереження анотацій...")

for split, files in splits.items():
    for img_file in tqdm(files, desc=f"{split}"):
        # Завантажити зображення через requests
        dst_path = os.path.join(OUTPUT_DIR, 'images', split, img_file)
        img_url = COCO_IMAGES_URL + img_file
        try:
            img_data = requests.get(img_url, timeout=10).content
            with open(dst_path, 'wb') as f:
                f.write(img_data)
        except Exception as e:
            print(f"⚠️ Не вдалося завантажити {img_file}: {e}")

        # записати анотацію
        label_file = img_file.replace('.jpg', '.txt')
        label_path = os.path.join(OUTPUT_DIR, 'labels', split, label_file)
        with open(label_path, 'w') as f:
            f.write('\n'.join(img_to_anns[img_file]))

print("\n✅ Готово. Новий збалансований датасет збережено в 'coco_balanced_dataset/'")
