from ultralytics import YOLO

if __name__ == "__main__":
    #加载模型
    model = YOLO(r"yolo11n.pt")
    #开启训练
    model.train(
        #训练“coco8”这个数据集
        data=r"args.yaml",
        #训练轮数
        epochs=100,
        imgsz=640,
        batch=8,
        cache="ram",
        workers=1,
        #增强参数
        hsv_h=0.02, hsv_s=0.8, hsv_v=0.5,  # 颜色抖动
        degrees=10.0,  # 旋转 ±10°
        translate=0.2, scale=0.6,  # 平移/缩放
        shear=5.0,  # 剪切
        flipud=0.2, fliplr=0.5,  # 翻转
        mosaic=1.0, mixup=0.2, cutmix=0.1,  # 拼接/混合
        copy_paste=0.1, erasing=0.4,  # 复制粘贴/随机擦除
        auto_augment="randaugment",  # 自动增强策略
    )