from ultralytics import YOLO

# === Завантаження базової моделі YOLOv8s ===
model = YOLO('G:/Course_four_2/КРБ/Project/YOLO/yolov8s.pt')

# === Навчання моделі ===
model.train(
    data='G:/Course_four_2/КРБ/Project/coco_balanced_dataset/data.yaml',  # Шлях до data.yaml
    epochs=25,                           # Кількість епох (можна змінити)
    imgsz=640,                           # Розмір зображень
    batch=8,                             # Batch size для CPU
    name='yolov8s_trein_coco_balanced_dataset',     # Назва сесії (папка з результатами)
    project='runs/detect',              # Головна папка з результатами
    exist_ok=True,                      # Не видавати помилку, якщо така папка вже існує
    device='cpu',                       # Навчання на CPU
    patience=10                         # Early stopping — зупинка при відсутності покращення

)



