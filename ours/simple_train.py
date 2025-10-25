from ultralytics import YOLO
from collections import defaultdict

f1_over_epoch = defaultdict(list) # dict:{int:list},class_id -> f1_list (over epoch)
def on_val_end(validator):
    metrics = validator.metrics
    box = metrics.box
    f1_list =  box.f1  # 各个类别预测的f1
    for class_id in range(len(f1_list)):
        f1_over_epoch[class_id].append(f1_list[class_id])

def simple_train():
    model = YOLO("pretrained_models/yolo11n.pt")  # load a pretrained model (recommended for training)
    model.train(data="african-wildlife.yaml", epochs=2, imgsz=640, val=True)

def train_with_callback():
    model = YOLO("pretrained_models/yolo11n.pt")  # load a pretrained model (recommended for training)
    # model.add_callback("on_val_end", on_val_end)
    results = model.train(data="african-wildlife.yaml", epochs=100, imgsz=640, val=True)
    # 保存训练结果
    # joblib.dump(f1_over_epoch,"error_f1.joblib")

if __name__ == "__main__":
    simple_train()






