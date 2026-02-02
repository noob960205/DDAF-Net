from ultralytics import YOLO

# 加载模型
# model = YOLO('./runs/detect/FLIR-DDAF/weights/best.pt')
model = YOLO('./runs/detect/M3FD-DDAF/weights/best.pt')


if __name__ == '__main__':
    # model structure
    # print(model.model)

    # metrics = model.val()
    # metrics = model.val(data='./yaml/data/FLIR-aligned-MM.yaml',  # 指定你的数据集配置文件
    metrics = model.val(data='./yaml/data/M3FD-MM.yaml',  # 指定你的数据集配置文件
                        save_txt=True,
                        save_conf=True,
                        # conf=0.5,  # 指定一个不同于默认的置-信度阈值来评估
                        # iou=0.6,  # 指定一个不同于默认的IoU阈值
                        device="cuda:2",
                        )
    print("\n================== Validation Results ==================")

    # 输出mAP指标
    print(f"\nmAP@50: {metrics.box.map50:.4f}")
    print(f"mAP@75: {metrics.box.map75:.4f}")
    print(f"mAP@50-95: {metrics.box.map:.4f}")
    # 平均精确率、召回率和F1
    print(f"\nPrecision (mean): {metrics.box.p.mean():.4f}")
    print(f"Recall (mean): {metrics.box.r.mean():.4f}")
    print(f"F1 Score (mean): {metrics.box.f1.mean():.4f}")
    # 输出每个类别的AP
    print("\nAPs per category (mAP@0.5:0.95):", metrics.box.maps)
    print(f"\nValidation complete. Results saved to: {metrics.save_dir}")