import sys

sys.path.append("../yolov5_in_tf2_keras")

import cv2
import os
import numpy as np
import random
import tensorflow as tf
from visual_ops import draw_bounding_box
from generate_coco_data import CoCoDataGenrator
from yolo import Yolo
from loss import ComputeLoss

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def main():
    # model_path = "h5模型路径, 默认在 ./logs/yolov5-tf-300.h5"
    model_path = "./logs/yolov_firi_modified-best.h5"
    # image_path = "提供你要测试的图片路径"
    image_path = "./data/y.jpg"
    result_img = "./data/predicts.jpg"
    # image_path = "./data/coco_2017_val_images/289343.jpg"
    # image_path = "./data/cat_dog_face_data/JPEGImages/Cats_Test849.jpg"
    # image_path = "./data/cat_dog_face_data/JPEGImages/Cats_Test214.jpg"
    image = cv2.imread(image_path)
    # 可以选择 ['5l', '5s', '5m', '5x'], 跟随训练
    image_shape = (512, 512, 3)
    # num_class = 91
    num_class = 3
    batch_size = 1

    # 这里anchor归一化到[0,1]区间
    anchors = (
        np.array(
            [
                [129, 16],
                [74, 36],
                [47, 70],
                [149, 39],
                [105, 72],
                [137, 62],
                [143, 69],
                [170, 62],
                [154, 75],
                [162, 80],
                [171, 87],
                [196, 100],
            ]
        )
        / image_shape[1]
    )
    anchor_masks = np.array(
        [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]], dtype=np.int8
    )
    anchors = np.array(anchors, dtype=np.float32)
    classes = ["defects-on-solarpanel", "Multi-Cell", "Single-Cell"]

    yolo = Yolo(
        model_path=model_path,
        image_shape=image_shape,
        batch_size=batch_size,
        num_class=num_class,
        is_training=False,
        anchors=anchors,
        anchor_masks=anchor_masks,
        strides=[4, 8, 16, 32],
        yolo_conf_threshold=0.75,
    )
    yolo.load_weights(model_path)

    # 预测结果: [nms_nums, (x1, y1, x2, y2, conf, cls)]
    predicts = yolo.predict(image)
    print(predicts)
    if predicts.shape[0]:
        pred_image = image.copy()
        for box_obj_cls in predicts[0]:
            if box_obj_cls[4] > 0.5:
                label = int(box_obj_cls[5])
                class_name = classes[label]
                xmin, ymin, xmax, ymax = box_obj_cls[:4]
                pred_image = draw_bounding_box(
                    pred_image,
                    class_name,
                    box_obj_cls[4],
                    int(xmin),
                    int(ymin),
                    int(xmax),
                    int(ymax),
                )
        cv2.imwrite(result_img, pred_image)
        # cv2.imshow("prediction", pred_image)
        # cv2.waitKey(0)


if __name__ == "__main__":
    tf.config.run_functions_eagerly(True)
    tf.data.experimental.enable_debug_mode()
    main()
