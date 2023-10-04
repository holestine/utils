from ultralytics import YOLO

YOLO_MODEL = 'yolov8n'
OBJECT     = 'object'
EXT        = 'pt'

model = YOLO('{}.{}'.format(YOLO_MODEL, EXT))

results = model.train(
    data = 'object_v8.yaml',
    imgsz = 640,
    epochs = 10,
    batch = 8,
    name = '{}-{}'.format(YOLO_MODEL, OBJECT)
)
