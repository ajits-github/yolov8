# from ultralytics import YOLO

# # Load a model
# # model = YOLO('yolov8n-seg.pt')  # load an official model
# model = YOLO('./runs/segment/train35/weights/best.pt', task="segment", mode="predict")  # load a custom model

# # Predict with the model
# # results = model('https://ultralytics.com/images/bus.jpg')  # predict on an image
# results = model('https://ultralytics.com/images/bus.jpg', source='G:\My Drive\Colab Notebooks\MUST\datasets\Teeth_WholeJaw_Ramina\RAMINA REHIMOVA0257.jpg', save=False, retina_masks=True, boxes=False, save_txt=True)  # predict on an image



from ultralytics import YOLO

model1 = ".\\runs\segment\\train35\\weights\\best.pt"
name = 'medium'
source="G:\My Drive\Colab Notebooks\MUST\datasets\Teeth_WholeJaw_Ramina\RAMINA REHIMOVA0257.jpg"

# proj_path = "/content/drive/MyDrive/Colab Notebooks/MUST/ultralytics-main"
# %cd '{proj_path}'

# model = YOLO('yolov8s-seg.pt')  # load a pretrained YOLOv8n segmentation model
# model = YOLO('yolov8s.pt') 
model = YOLO(model1)
# model.predict(source=source, task="segment", retina_masks=True, boxes=False, project='/content', save_txt=True, save_conf=True, name=name)
results = model(source=source, save=False, retina_masks=True, boxes=False, save_txt=True)  # predict on an image
