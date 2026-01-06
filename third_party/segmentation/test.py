from ultralytics import ASSETS, SAM, YOLO, FastSAM

# Profile SAM2-t, SAM2-b, SAM-b, MobileSAM
for file in ["sam_b.pt", "sam2_b.pt", "sam2_t.pt", "mobile_sam.pt"]:
    model = SAM(file)
    model.info()
    model(ASSETS)

# Profile FastSAM-s
model = FastSAM("FastSAM-s.pt")
model.info()
model(ASSETS)

# Profile YOLO models
for file_name in ["yolov8n-seg.pt", "yolo11n-seg.pt"]:
    model = YOLO(file_name)
    model.info()
    model(ASSETS)

# from ultralytics import FastSAM, SAN

# # 定义路径变量，方便复用
# model_path = "/home/dell/sunyulong/GICI-IM/third_party/segmentation/FastSAM-s.pt"
# image_path = "/home/dell/sunyulong/GICI-IM/third_party/segmentation/framew0006.jpg"

# # Load a model
# model = FastSAM(model_path)

# # Display model information (optional)
# model.info()

# # Run inference
# results = model(image_path, texts="a black car")
# for result in results:
#     result.save(filename="result_default_mobile.jpg")  # 保存结果 

# # 1. Run inference with bboxes prompt
# # 注意：这里添加了 save=True，或者手动处理 results
# results = model(image_path, bboxes=[100, 100, 200, 200])
# for result in results:
#     result.save(filename="result_bbox.jpg")  # 保存结果

# # 2. Run inference with single point
# # 注意：必须传入 image_path
# results = model(image_path, points=[900, 370], labels=[1])
# for result in results:
#     result.save(filename="result_single_point.jpg")

# # 3. Run inference with multiple points
# results = model(image_path, points=[[400, 370], [900, 370]], labels=[1, 1])
# for result in results:
#     result.save(filename="result_multi_points.jpg")

# # 4. Run inference with multiple points prompt per object
# results = model(image_path, points=[[[400, 370], [900, 370]]], labels=[[1, 1]])
# for result in results:
#     result.save(filename="result_multi_objects.jpg")

# # 5. Run inference with negative points prompt
# results = model(image_path, points=[[[400, 370], [900, 370]]], labels=[[1, 0]])
# for result in results:
#     result.save(filename="result_negative_points.jpg")

# from ultralytics import SAM

# # Load the model
# model = SAM("mobile_sam.pt")

# # Predict a segment based on a single point prompt
# model.predict("ultralytics/assets/zidane.jpg", points=[900, 370], labels=[1])

# # Predict multiple segments based on multiple points prompt
# model.predict("ultralytics/assets/zidane.jpg", points=[[400, 370], [900, 370]], labels=[1, 1])

# # Predict a segment based on multiple points prompt per object
# model.predict("ultralytics/assets/zidane.jpg", points=[[[400, 370], [900, 370]]], labels=[[1, 1]])

# # Predict a segment using both positive and negative prompts.
# model.predict("ultralytics/assets/zidane.jpg", points=[[[400, 370], [900, 370]]], labels=[[1, 0]])