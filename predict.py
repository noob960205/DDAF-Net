# -*- coding=utf-8 -*-
# author:"GH"
# date:2025/10/11 09:14
# -*- coding: utf-8 -*-
# author: GH
# date: 2025/10/11

import cv2
import numpy as np
from ultralytics import YOLO


def load_model(model_path):
    """
    根据传入路径加载 YOLO 模型
    :param model_path: 模型权重路径，例如 "./runs/detect/M3FD-DHAFNet/weights/best.pt"
    :return: 已加载的 YOLO 模型对象
    """
    print(f"✅ 正在加载模型: {model_path}")
    model = YOLO(model_path)
    return model


def predict(model, model_type="multi", rgb_path=None, ir_path=None,
            save=True, imgsz=640, conf=0.5, device=0, iou=0.5):
    """
    通用预测函数，支持多模态或单模态
    :param model: 已加载的 YOLO 模型对象
    :param model_type: "multi" 表示多模态（RGB+IR），"single" 表示单模态
    :param rgb_path: RGB 图像路径
    :param ir_path: IR 图像路径
    """
    if model_type == "multi":
        if rgb_path is None or ir_path is None:
            raise ValueError("多模态预测必须同时提供 rgb_path 和 ir_path。")

        img_rgb = cv2.imread(rgb_path)
        img_ir = cv2.imread(ir_path)

        if img_rgb is None or img_ir is None:
            raise FileNotFoundError("无法读取指定的 RGB 或 IR 图像，请检查路径。")

        # notice：拼接顺序，ultralytics.data.base.BaseDataset.load_multi_image中im = np.dstack((im_rgb, im_ir))
        stacked_image = np.dstack((img_rgb, img_ir))  # 通道堆叠 (H, W, 6)
        source = stacked_image
        print("🔹 正在进行多模态推理...")

    elif model_type == "single":
        # 单模态可为 RGB 或 IR
        if rgb_path:
            source = rgb_path
            print("🔹 正在进行单模态 RGB 推理...")
        elif ir_path:
            source = ir_path
            print("🔹 正在进行单模态 IR 推理...")
        else:
            raise ValueError("单模态预测必须提供 rgb_path 或 ir_path。")

    else:
        raise ValueError(f"未知的 model_type 参数: {model_type}，仅支持 'multi' 或 'single'。")

    results = model.predict(
        source=source,
        save=save,
        imgsz=imgsz,
        conf=conf,
        device=device,
        iou=iou
    )

    print("✅ 推理完成。")
    return results


if __name__ == "__main__":
    # 加载模型
    model = load_model("./runs/detect/FLIR-DDAF/weights/best.pt")
    # 单模态推理（IR）
    # model = load_model("./runs/detect/M3FD-IR/weights/best.pt")

    # 多模态推理
    results = predict(
        model=model,
        model_type="multi",
        rgb_path="./datasets/FLIR-aligned/imageRGB/test/09041.jpg",
        ir_path="./datasets/FLIR-aligned/imageIR/test/09041.jpg"
    )

    # results = predict(model=model, model_type="single", ir_path="./datasets/M3FD/imageIR/test/00079.png")

    # 遍历每张图片的检测结果
    for i, result in enumerate(results):
        print(f"图像 {i} 的检测结果：")

        boxes = result.boxes  # Boxes对象
        names = result.names  # 类别名映射表

        for box in boxes:
            # xyxy格式坐标
            xyxy = box.xyxy[0].cpu().numpy()  # 例如 [x1, y1, x2, y2]
            conf = float(box.conf[0])  # 置信度
            cls_id = int(box.cls[0])  # 类别ID
            cls_name = names[cls_id]  # 类别名称

            print(f"类别: {cls_name} | 置信度: {conf:.2f} | 框: {xyxy}")