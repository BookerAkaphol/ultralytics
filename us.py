from ultralytics import YOLO


model = YOLO("sss.pt")


results = model.predict(source="0", conf=0.30)
