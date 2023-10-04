from ultralytics import YOLO
from pathlib import Path
import cv2

best_yolo_model = 'runs/detect/yolov8n-coupler/weights/best.pt'
image_folder = 'data/unlabeled/images/'
image_ext = '*.jpeg'

def show_image(path, type='unknown', boxes=[]):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    
    for box in boxes:
        x1 = int(box[0])
        y1 = int(box[1])
        x2 = int(box[2])
        y2 = int(box[3])
        img = cv2.rectangle(img, (x1,y1), (x2,y2), 127, 3)
    
    img = cv2.resize(img,(0, 0),fx=0.5, fy=0.5, interpolation = cv2.INTER_AREA)
    
    # show image
    cv2.imshow(type, img)
    cv2.moveWindow(type, 10, 10)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

missing_couplers  = []
multiple_couplers = []
low_conf_couplers = []

model = YOLO(best_yolo_model)
test_images = [str(x) for x in list(Path(image_folder).glob(image_ext))]
chunk_size = int(len(test_images) / 16)

for i in range(0, len(test_images), chunk_size):
    results = model.predict(test_images[i:i + chunk_size])
    for res in results:
        if len(res.boxes.conf) == 0:
            missing_couplers.append(Path(res.path).stem)
            show_image(res.path, 'Missing')
        elif len(res.boxes.conf) > 1:
            multiple_couplers.append(Path(res.path).stem)
            show_image(res.path, 'Multiple', res.boxes.xyxy.detach().cpu().numpy())
        elif min(res.boxes.conf) < 0.8:
            low_conf_couplers.append(Path(res.path).stem)
            show_image(res.path, 'Low', res.boxes.xyxy.detach().cpu().numpy())

def print_defects(defect_name, defects):
    print(defect_name)
    for defect in defects:
        print(defect)
    print('\n')

print_defects('\nMissing couplers',      missing_couplers )
print_defects('Multiple couplers',       multiple_couplers)
print_defects('Low confidence couplers', low_conf_couplers)

