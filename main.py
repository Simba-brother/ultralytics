from ultralytics import YOLO
from ultralytics import settings

def on_val_end(validator):
    """Print trainer metrics and loss details after each epoch is trained."""
    # 获得每次验证的类指标
    metrics = validator.metrics
    box = metrics.box
    f1_list =  box.f1  # 各个类别预测的f1

# Load a model
model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)
# model.add_callback("on_train_epoch_end", on_train_epoch_end)
model.add_callback("on_val_end", on_val_end)
# Train the model
results = model.train(
    data="african-wildlife.yaml", 
    epochs=2, 
    imgsz=640)
