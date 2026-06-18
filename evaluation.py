

from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import os

# FUNCTION TO EVALUATE MODEL


def evaluate_model(model_path, data_yaml, model_name):

    print("\n================================================")
    print(f" EVALUATING : {model_name}")
    print("================================================\n")

    model = YOLO(model_path)


    metrics = model.val(
        data=data_yaml,
        imgsz=640,
        conf=0.25,
        iou=0.5,
        split='test',
        save_json=True
    )


    print(f"\n===== {model_name} RESULTS =====\n")

    print("Precision :", round(metrics.box.mp, 4))
    print("Recall    :", round(metrics.box.mr, 4))
    print("mAP@50    :", round(metrics.box.map50, 4))
    print("mAP@50-95 :", round(metrics.box.map, 4))


    results_dir = metrics.save_dir

    print("\nResults saved in:")
    print(results_dir)

    graphs = [
        "confusion_matrix.png",
        "PR_curve.png",
        "P_curve.png",
        "R_curve.png",
        "F1_curve.png",
        "results.png"
    ]

    for graph in graphs:

        graph_path = os.path.join(results_dir, graph)

        if os.path.exists(graph_path):

            img = cv2.imread(graph_path)

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            plt.figure(figsize=(10, 8))

            plt.imshow(img)

            plt.title(f"{model_name} - {graph}")

            plt.axis("off")

            plt.show()

        else:

            print(f"{graph} not found.")

# HELMET MODEL EVALUATION

helmet_model_path = (
    "Motorcycle Helmet and License plate detection.v6-datasetv6.yolov8/helmetdetection/Weights/best.pt"
)

helmet_data_yaml = (
    "Motorcycle Helmet and License plate detection.v6-datasetv6.yolov8/data.yaml"
)

evaluate_model(
    helmet_model_path,
    helmet_data_yaml,
    "Helmet Detection Model"
)

# LICENSE PLATE MODEL EVALUATION

plate_model_path = (
    "License Plate Recognition.v13i.yolov8/helmet_model_final/liceplate/Weights/best.pt"
)

plate_data_yaml = (
    "License Plate Recognition.v13i.yolov8/data.yaml"
)

evaluate_model(
    plate_model_path,
    plate_data_yaml,
    "License Plate Detection Model"
)
