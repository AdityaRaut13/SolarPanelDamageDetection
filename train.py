import os
import tqdm
import numpy as np
import random
import tensorflow as tf
from generate_coco_data import CoCoDataGenrator
from visual_ops import draw_bounding_box
from yolo import Yolo
from loss import ComputeLoss
from val import val
from layers import nms


def main():
    epochs = 150
    log_dir = "./logs/"
    # log_dir = "/kaggle/working/logs/"
    # 可以选择 ['5l', '5s', '5m', '5x']

    image_shape = (512, 512, 3)
    # num_class = 91
    num_class = 3
    batch_size = 3
    # -1表示全部数据参与训练
    train_img_nums = -1
    # train_coco_json = "/kaggle/input/solar-panel-damage/train/_annotations.coco.json"
    # val_coco_json = "/kaggle/input/solar-panel-damage/test/_annotations.coco.json"
    # model_path = "/kaggle/input/logs/yolov_firi_modified-last.h5"
    train_coco_json = "../newDataset/train/_annotations.coco.json"
    val_coco_json = "../newDataset/test/_annotations.coco.json"
    model_path = "./logs/yolov_firi_modified-last.h5"
    classes = [
        "SolarPanel-Damage",
        "Multi-Cell",
        "Single-Cell",
    ]
    # 这里anchor归一化到[0,1]区间

    # anchors = (
    # np.array(
    # [
    # [129, 16],
    # [74, 36],
    # [47, 70],
    # [149, 39],
    # [105, 72],
    # [137, 62],
    # [143, 69],
    # [170, 62],
    # [154, 75],
    # [162, 80],
    # [171, 87],
    # [196, 100],
    # ]
    # )
    # / image_shape[1]
    # )

    anchors = (
        np.array(
            [
                [18, 88],
                [109, 16],
                [45, 73],
                [75, 45],
                [60, 155],
                [90, 149],
                [157, 92],
                [104, 166],
                [177, 116],
                [125, 191],
                [95, 512],
                [512, 138],
            ]
        )
        / image_shape[1]
    )

    anchor_masks = np.array(
        [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]], dtype=np.int8
    )
    anchors = np.array(anchors, dtype=np.float32)
    summary_writer = tf.summary.create_file_writer(log_dir)
    # data generator
    coco_data = CoCoDataGenrator(
        coco_annotation_file=train_coco_json,
        train_img_nums=train_img_nums,
        img_shape=image_shape,
        batch_size=batch_size,
        include_mask=False,
        include_crowd=False,
        include_keypoint=False,
        need_down_image=False,
        using_argument=True,
    )
    # 验证集
    val_coco_data = CoCoDataGenrator(
        coco_annotation_file=val_coco_json,
        train_img_nums=-1,
        img_shape=image_shape,
        batch_size=batch_size,
        include_mask=False,
        include_crowd=False,
        include_keypoint=False,
        need_down_image=False,
        using_argument=False,
    )

    yolo = Yolo(
        model_path=model_path,
        image_shape=image_shape,
        batch_size=batch_size,
        num_class=num_class,
        is_training=True,
        anchors=anchors,
        anchor_masks=anchor_masks,
        strides=[4, 8, 16, 32],
        use_net="v2",
    )
    yolo.yolo_firi_modified.summary(line_length=200)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001)

    loss_fn = ComputeLoss(
        image_shape=image_shape,
        anchors=anchors,
        anchor_masks=anchor_masks,
        num_class=num_class,
        anchor_ratio_thres=4,
        only_best_anchor=False,
        balanced_rate=15,
        iou_ignore_thres=0.5,
        layer_balance=[3.0, 3.0, 3.0, 3.0],
    )

    pre_mAP = 9.7694510073729e-05
    # data = coco_data.next_batch()
    for epoch in range(epochs):
        train_progress_bar = tqdm.tqdm(
            range(coco_data.total_batch_size),
            desc="train epoch {}/{}".format(epoch, epochs - 1),
            ncols=100,
        )
        for batch in train_progress_bar:
            with tf.GradientTape() as tape:
                data = coco_data.next_batch()
                valid_nums = data["valid_nums"]
                gt_imgs = np.array(data["imgs"] / 255.0, dtype=np.float32)
                gt_boxes = np.array(data["bboxes"] / image_shape[1], dtype=np.float32)
                gt_classes = data["labels"]

                # print("-------epoch {}, step {}, total step {}--------".format(epoch, batch,
                #                                                                epoch * coco_data.total_batch_size + batch))
                # print("current data index: ",
                #       coco_data.img_ids[(coco_data.current_batch_index - 1) * coco_data.batch_size:
                #                         coco_data.current_batch_index * coco_data.batch_size])
                # for i, nums in enumerate(valid_nums):
                #     print("gt boxes: ", gt_boxes[i, :nums, :] * image_shape[0])
                #     print("gt classes: ", gt_classes[i, :nums])

                yolo_preds = yolo.yolo_firi_modified(gt_imgs, training=True)
                loss_xy, loss_wh, loss_box, loss_obj, loss_cls = loss_fn(
                    yolo_preds, gt_boxes, gt_classes
                )

                total_loss = loss_box + loss_obj + loss_cls
                train_progress_bar.set_postfix(
                    ordered_dict={"loss": "{:.5f}".format(total_loss)}
                )

                grad = tape.gradient(
                    total_loss, yolo.yolo_firi_modified.trainable_variables
                )
                optimizer.apply_gradients(
                    zip(grad, yolo.yolo_firi_modified.trainable_variables)
                )

                # Scalar
                with summary_writer.as_default():
                    tf.summary.scalar(
                        "loss/box_loss",
                        loss_box,
                        step=epoch * coco_data.total_batch_size + batch,
                    )
                    tf.summary.scalar(
                        "loss/object_loss",
                        loss_obj,
                        step=epoch * coco_data.total_batch_size + batch,
                    )
                    tf.summary.scalar(
                        "loss/class_loss",
                        loss_cls,
                        step=epoch * coco_data.total_batch_size + batch,
                    )
                    tf.summary.scalar(
                        "loss/total_loss",
                        total_loss,
                        step=epoch * coco_data.total_batch_size + batch,
                    )

                # image, 只拿每个batch的其中一张
                random_one = random.choice(range(batch_size))
                # gt
                gt_img = gt_imgs[random_one].copy() * 255
                gt_box = gt_boxes[random_one] * image_shape[1]
                gt_class = gt_classes[random_one]
                non_zero_ids = np.where(np.sum(gt_box, axis=-1))[0]
                for i in non_zero_ids:
                    cls = gt_class[i]
                    class_name = coco_data.coco.cats[cls]["name"]
                    xmin, ymin, xmax, ymax = gt_box[i]
                    # print(xmin, ymin, xmax, ymax)
                    gt_img = draw_bounding_box(
                        gt_img,
                        class_name,
                        cls,
                        int(xmin),
                        int(ymin),
                        int(xmax),
                        int(ymax),
                    )

                # pred, 同样只拿第一个batch的pred
                pred_img = gt_imgs[random_one].copy() * 255
                yolo_head_output = yolo.yolo_head(yolo_preds, is_training=False)
                nms_output = nms(image_shape, yolo_head_output.numpy(), iou_thres=0.3)
                if len(nms_output) == batch_size:
                    nms_output = nms_output[random_one]
                    for box_obj_cls in nms_output:
                        if box_obj_cls[4] > 0.5:
                            label = int(box_obj_cls[5])
                            if coco_data.coco.cats.get(label):
                                class_name = coco_data.coco.cats[label]["name"]
                                # class_name = classes[label]
                                xmin, ymin, xmax, ymax = box_obj_cls[:4]
                                pred_img = draw_bounding_box(
                                    pred_img,
                                    class_name,
                                    box_obj_cls[4],
                                    int(xmin),
                                    int(ymin),
                                    int(xmax),
                                    int(ymax),
                                )

                concat_imgs = tf.concat(
                    [gt_img[:, :, ::-1], pred_img[:, :, ::-1]], axis=1
                )
                summ_imgs = tf.expand_dims(concat_imgs, 0)
                summ_imgs = tf.cast(summ_imgs, dtype=tf.uint8)
                with summary_writer.as_default():
                    tf.summary.image(
                        "imgs/gt,pred,epoch{}".format(epoch),
                        summ_imgs,
                        step=epoch * coco_data.total_batch_size + batch,
                    )
        # 这里计算一下训练集的mAP
        val(
            model=yolo,
            val_data_generator=coco_data,
            classes=classes,
            desc="training dataset val",
        )
        # 这里计算验证集的mAP
        mAP50, mAP, final_df = val(
            model=yolo,
            val_data_generator=val_coco_data,
            classes=classes,
            desc="val dataset val",
        )
        if mAP > pre_mAP:
            pre_mAP = mAP
            yolo.yolo_firi_modified.save_weights(
                log_dir + "/yolov_firi_modified-best.h5"
            )
            print(
                "save {}/yolov_firi_modified-best.h5 best weight with {} mAP.".format(
                    log_dir, mAP
                )
            )
        yolo.yolo_firi_modified.save_weights(log_dir + "/yolov_firi_modified-last.h5")
        print(
            "save {}/yolov_firi_modified-last.h5 last weights at epoch {}.".format(
                log_dir, epoch
            )
        )


if __name__ == "__main__":
    import sys

    with open("./output.txt", "w") as sys.stdout:
        tf.config.run_functions_eagerly(True)
        gpus = tf.config.experimental.list_physical_devices("GPU")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        tf.data.experimental.enable_debug_mode()
        main()
