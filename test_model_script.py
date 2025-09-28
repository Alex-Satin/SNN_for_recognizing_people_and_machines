import os
import pandas as pd
import matplotlib.pyplot as plt

# Назва моделі та шлях до її CSV-файлу
model_name = 'YOLOv8n'
csv_path = 'runs/detect/yolov8n_custom/results.csv'

# Папка для збереження результатів
base_save_dir = os.path.join(os.path.dirname(__file__), 'yolo_results', model_name)
os.makedirs(base_save_dir, exist_ok=True)

# Перевірка наявності файлу
if not os.path.exists(csv_path):
    print(f"[ERROR] Не знайдено файл: {csv_path}")
    exit()

df = pd.read_csv(csv_path)
if df.empty:
    print(f"[WARNING] CSV-файл {csv_path} порожній.")
    exit()

epochs = df['epoch']

# --- Графіки ---
# 1. Втрати
plt.figure(figsize=(10, 5))
plt.plot(epochs, df['train/box_loss'], label='Box Loss')
plt.plot(epochs, df['train/cls_loss'], label='Class Loss')
plt.plot(epochs, df['train/dfl_loss'], label='DFL Loss')
plt.title(f'{model_name} - Втрати по епохах')
plt.xlabel('Епоха')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(base_save_dir, f'{model_name}_losses.png'))
plt.close()

# Precision окремо
plt.figure(figsize=(10, 5))
plt.plot(epochs, df['metrics/precision(B)'], color='blue')
plt.title(f'{model_name} - Precision по епохах')
plt.xlabel('Епоха')
plt.ylabel('Precision')
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(base_save_dir, f'{model_name}_precision_only.png'))
plt.close()

# Recall окремо
plt.figure(figsize=(10, 5))
plt.plot(epochs, df['metrics/recall(B)'], color='green')
plt.title(f'{model_name} - Recall по епохах')
plt.xlabel('Епоха')
plt.ylabel('Recall')
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(base_save_dir, f'{model_name}_recall_only.png'))
plt.close()


# 3. mAP
plt.figure(figsize=(10, 5))
plt.plot(epochs, df['metrics/mAP50(B)'], label='mAP@0.5')
plt.plot(epochs, df['metrics/mAP50-95(B)'], label='mAP@0.5:0.95')
plt.title(f'{model_name} - mAP по епохах')
plt.xlabel('Епоха')
plt.ylabel('mAP')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(base_save_dir, f'{model_name}_map.png'))
plt.close()

# --- Таблиця метрик останньої епохи ---
last = df.iloc[-1]
metrics_table = {
    "Epoch": int(last["epoch"]),
    "Precision": round(last['metrics/precision(B)'], 4),
    "Recall": round(last['metrics/recall(B)'], 4),
    "mAP@0.5": round(last['metrics/mAP50(B)'], 4),
    "mAP@0.5:0.95": round(last['metrics/mAP50-95(B)'], 4),
    "IoU loss(val/box_loss)": round(last.get('val/box_loss', 0), 4)
}

txt_path = os.path.join(base_save_dir, f"{model_name}_metrics.txt")
with open(txt_path, "w", encoding="utf-8") as f:
    f.write(f"=== Підсумкові метрики для моделі {model_name} ===\n\n")
    for k, v in metrics_table.items():
        f.write(f"{k}: {v}\n")

print(f"[INFO] Метрики для {model_name} збережено в {txt_path}")

# --- Збереження таблиці метрик по епохах ---
matrix_csv_path = os.path.join(base_save_dir, f"{model_name}_metrics_per_epoch.csv")
df_metrics = df[[
    'epoch', 'metrics/precision(B)', 'metrics/recall(B)',
    'metrics/mAP50(B)', 'metrics/mAP50-95(B)',
    'val/box_loss'
]]
df_metrics.to_csv(matrix_csv_path, index=False)
print(f"[INFO] Матриця метрик збережена: {matrix_csv_path}")
