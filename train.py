# -*- coding: utf-8 -*-
# author: GH
# date: 2025/10/11
"""
通用 YOLO 训练脚本
通过修改下方参数区直接配置不同模型的训练任务
"""

from ultralytics import YOLO


def train_yolo(model_yaml, data_yaml, name,
               device="cuda:0", epochs=300, batch=8,
               imgsz=640, amp=False, lr0=0.001,
               optimizer='AdamW'):
    """
    通用 YOLO 训练函数
    :param model_yaml: 模型结构 YAML 文件路径
    :param data_yaml: 数据集 YAML 文件路径
    :param name: 训练任务名称（保存目录名）
    :param device: 使用的 GPU 或 CPU
    :param epochs: 训练轮数
    :param batch: batch 大小
     - RGBorIR: n=32, m=16, x=8
     - EarlyFusion: n=32, m=16, x=8
     - TwoStream: n=32, m=16, x=8
     - Siamese: n=32, m=16, x=8
     - DHAFNet: n=16, m=8, x=4
    :param imgsz: 输入图像大小
    :param amp: 是否启用自动混合精度（多GPU训练需禁用）
    :param lr0: 初始学习率
    :param optimizer: 优化器类型
    """
    print(f"✅ 正在加载模型结构: {model_yaml}")
    model = YOLO(model_yaml)

    print(f"🚀 开始训练: {name}")
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        name=name,
        device=device,
        amp=amp,
        lr0=lr0,
        optimizer=optimizer
    )

    print("✅ 训练完成。")
    return results


if __name__ == "__main__":
    # ================================================
    # 🔧【配置区】修改这里即可控制训练
    # ================================================

    # 模型配置文件路径（选择一个）
    # model_yaml = "yaml/model/yolov8x-RGBorIR.yaml"
    # model_yaml = "yaml/model/yolov8x-EarlyFusion.yaml"
    # model_yaml = "yaml/model/yolov8x-TwoStream.yaml"
    # model_yaml = "yaml/model/yolov8x-Siamese.yaml"
    model_yaml = "yaml/model/yolov8x-DDAF.yaml"

    # 数据集配置文件路径
    # data_yaml = "./yaml/data/M3FD-IR.yaml"
    data_yaml = "./yaml/data/LLVIP-MM.yaml"

    # 训练任务名称（用于保存目录）
    name = "LLVIP-DDAF"
    # 设备、训练参数
    device = "cuda:3"
    epochs = 300
    batch = 4
    imgsz = 640
    amp = False
    lr0 = 0.001
    optimizer = "AdamW"

    # ================================================
    # 🚀 执行训练
    # ================================================
    train_yolo(
        model_yaml=model_yaml,
        data_yaml=data_yaml,
        name=name,
        device=device,
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        amp=amp,
        lr0=lr0,
        optimizer=optimizer
    )
