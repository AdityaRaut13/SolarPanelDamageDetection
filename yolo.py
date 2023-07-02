import sys


import os
import numpy as np
import cv2
import tensorflow as tf
from layers import nms, YoloHead

from keras.layers import Reshape, Concatenate, Softmax, Input
from keras.models import Model
from yolo_firi_modified_small import Yolo_FIRI_Modified
from yolo_firi_modified_large import Yolo_FIRI_Modified_Large
from yolo_firi_modified_v2 import Yolo_FIRI_Modified_v2


class Yolo:
    def __init__(
        self,
        num_class,
        anchors,
        anchor_masks,
        image_shape=[640, 640, 3],
        is_training=True,
        batch_size=5,
        strides=[8, 16, 32],
        anchors_per_location=3,
        yolo_max_boxes=100,
        yolo_iou_threshold=0.3,
        yolo_conf_threshold=0.5,
        model_path=None,
        use_net="small",
    ):
        self.image_shape = image_shape
        self.is_training = is_training
        self.batch_size = batch_size
        self.strides = strides
        self.anchors_per_location = anchors_per_location
        self.yolo_max_boxes = yolo_max_boxes
        self.yolo_iou_threshold = yolo_iou_threshold
        self.yolo_conf_threshold = yolo_conf_threshold
        self.model_path = model_path

        self.num_class = num_class
        self.anchors = anchors
        self.anchor_masks = anchor_masks
        if use_net == "large":
            self.base_net = Yolo_FIRI_Modified_Large(
                image_shape=self.image_shape,
                batch_size=self.batch_size,
                num_class=self.num_class,
                anchors_per_location=self.anchors_per_location,
            ).build_graph()
        elif use_net == "small":
            self.base_net = Yolo_FIRI_Modified(
                image_shape=self.image_shape,
                batch_size=self.batch_size,
                num_class=self.num_class,
                anchors_per_location=self.anchors_per_location,
            ).build_graph()
        elif use_net == "v2":
            self.base_net = Yolo_FIRI_Modified_v2(
                image_shape=self.image_shape,
                batch_size=self.batch_size,
                num_class=self.num_class,
                anchors_per_location=self.anchors_per_location,
            ).build_graph()
        self.grid = []
        self.anchor_grid = []
        self.yolo_firi_modified = self.build_graph()
        if not is_training:
            assert model_path, "Inference mode need the model_path!"
            assert os.path.isfile(model_path), "Can't find the model weight file!"
            self.yolo_firi_modified.load_weights(model_path, by_name=True)
            print("loading model weight from {}".format(model_path))

    def load_weights(self, model_path, by_name=True, exclude=None):
        import h5py
        from tensorflow.python.keras.saving import hdf5_format

        if exclude:
            by_name = True

        if h5py is None:
            raise ImportError("`load_weights` requires h5py.")
        with h5py.File(model_path, mode="r") as f:
            if "layer_names" not in f.attrs and "model_weights" in f:
                f = f["model_weights"]

            # In multi-GPU training, we wrap the model. Get layers
            # of the inner model because they have the weights.
            layers = (
                self.yolo_firi_modified.inner_model.layers
                if hasattr(self.yolo_firi_modified, "inner_model")
                else self.yolo_firi_modified.layers
            )

            # Exclude some layers
            if exclude:
                layers = filter(lambda l: l.name not in exclude, layers)

            if by_name:
                hdf5_format.load_weights_from_hdf5_group_by_name(f, layers)
            else:
                hdf5_format.load_weights_from_hdf5_group(f, layers)

    def yolo_head(self, features, is_training):
        """yolo最后输出层
        :param features:
        :return: train mode: [[batch, h, w, num_anchors_per_layer, num_class + 5], [...], [...]]
                 infer mode: [batch, -1, num_class + 5]
        """
        # num_anchors_per_layer = len(self.anchors[0])
        detect_res = []
        for i, pred in enumerate(features):
            if is_training:
                detect_res.append(pred)
                continue
            f_shape = tf.shape(pred)
            if len(self.grid) < self.anchor_masks.shape[0]:
                grid, anchor_grid = self._make_grid(f_shape[1], f_shape[2], i)
                self.grid.append(grid)
                self.anchor_grid.append(anchor_grid)

            # 这里把输出的值域从[0,1]调整到[0, image_shape]
            pred_xy = (
                tf.sigmoid(pred[..., 0:2]) * 2.0 - 0.5 + self.grid[i]
            ) * self.strides[i]
            pred_wh = (tf.sigmoid(pred[..., 2:4]) * 2) ** 2 * self.anchor_grid[i]
            # print(self.grid)
            pred_obj = tf.sigmoid(pred[..., 4:5])
            pred_cls = Softmax()(pred[..., 5:])
            cur_layer_pred_res = Concatenate(axis=-1)(
                [pred_xy, pred_wh, pred_obj, pred_cls]
            )

            # cur_layer_pred_res = tf.reshape(cur_layer_pred_res, [self.batch_size, -1, self.num_class + 5])
            cur_layer_pred_res = Reshape([-1, self.num_class + 5])(cur_layer_pred_res)
            detect_res.append(cur_layer_pred_res)
        return detect_res if is_training else tf.concat(detect_res, axis=1)

    def _make_grid(self, h, w, i):
        cur_layer_anchors = self.anchors[self.anchor_masks[i]] * np.array(
            [[self.image_shape[1], self.image_shape[0]]]
        )
        num_anchors_per_layer = len(cur_layer_anchors)
        yv, xv = tf.meshgrid(tf.range(h), tf.range(w))
        grid = tf.stack((xv, yv), axis=2)
        # 用来计算中心点的grid cell左上角坐标
        grid = tf.tile(
            tf.reshape(grid, [1, h, w, 1, 2]), [1, 1, 1, num_anchors_per_layer, 1]
        )
        grid = tf.cast(grid, tf.float32)
        # anchor_grid = tf.reshape(cur_layer_anchors * self.strides[i], [1, 1, 1, num_anchors_per_layer, 2])
        anchor_grid = tf.reshape(cur_layer_anchors, [1, 1, 1, num_anchors_per_layer, 2])
        # 用来计算宽高的anchor w/h
        anchor_grid = tf.tile(anchor_grid, [1, h, w, 1, 1])
        anchor_grid = tf.cast(anchor_grid, tf.float32)

        return grid, anchor_grid

    def build_graph(self):
        # inputs = tf.keras.layers.Input(shape=self.image_shape, batch_size=self.batch_size)
        inputs = Input(shape=self.image_shape, batch_size=self.batch_size)
        yolo_body_outputs = self.base_net(inputs)
        tf.keras.utils.plot_model(
            self.base_net, to_file="model.png", expand_nested=True
        )

        # outputs = self.yolo_head(yolo_body_outputs, is_training=is_training)
        # outputs = self.yolo_head(yolo_body_outputs, is_training=True)
        outputs = YoloHead(
            image_shape=self.image_shape,
            num_class=self.num_class,
            is_training=self.is_training,
            strides=self.strides,
            anchors=self.anchors,
            anchors_masks=self.anchor_masks,
        )(yolo_body_outputs)
        # if not self.is_training:
        #     outputs = self.nms(outputs, iou_thres=self.yolo_iou_threshold, conf_thres=self.yolo_conf_threshold)
        # model = tf.keras.models.Model(inputs=inputs, outputs=yolo_body_outputs)
        model = Model(inputs=inputs, outputs=outputs)
        tf.keras.utils.plot_model(
            model, to_file="model_abstraction.png", expand_nested=True
        )
        return model

    def predict(self, images, image_need_resize=True, resize_to_origin=True):
        """预测
           预测模式下实例化类: is_training=False, weights_path=, batch_size跟随输入建议1, image_shape跟随训练模式,不做调整
        :param images: [batch, h, w, c] or [h, w, c]
        :return [[nms_nums, (x1, y1, x2, y2, conf, cls)], [...], [...], ...]
        """

        if len(np.shape(images)) <= 3:
            images = [images]
            self.batch_size = 1

        final_outputs = []
        for i, im in enumerate(images):
            if image_need_resize:
                im_shape = np.shape(im)
                im_size_max = np.max(im_shape[0:2])
                im_scale = float(self.image_shape[0]) / float(im_size_max)

                # resize原始图片
                im_resize = cv2.resize(
                    im,
                    None,
                    None,
                    fx=im_scale,
                    fy=im_scale,
                    interpolation=cv2.INTER_LINEAR,
                )
                im_resize_shape = np.shape(im_resize)
                im_blob = np.zeros(self.image_shape, dtype=np.float32)
                im_blob[0 : im_resize_shape[0], 0 : im_resize_shape[1], :] = im_resize
                inputs = np.array([im_blob], dtype=np.float32) / 255.0
            else:
                inputs = np.array([im], dtype=np.float32) / 255.0

            # 预测, [batch, -1, num_class + 5]
            # outputs = self.yolov5.predict(inputs)
            outputs = self.yolo_firi_modified.predict(inputs)
            nms_outputs = nms(self.image_shape, outputs)
            # nms_outputs = self.nms(outputs.numpy())
            # print(nms_outputs.shape)
            # if not nms_outputs.shape[0]:
            #     continue
            if not nms_outputs:
                continue
            nms_outputs = np.array(nms_outputs[0], dtype=np.float32)

            # resize回原图大小
            if resize_to_origin:
                boxes = nms_outputs[:, :4]
                b0 = np.maximum(np.minimum(boxes[:, 0] / im_scale, im_shape[1] - 1), 0)
                b1 = np.maximum(np.minimum(boxes[:, 1] / im_scale, im_shape[0] - 1), 0)
                b2 = np.maximum(np.minimum(boxes[:, 2] / im_scale, im_shape[1] - 1), 0)
                b3 = np.maximum(np.minimum(boxes[:, 3] / im_scale, im_shape[0] - 1), 0)
                origin_boxes = np.stack([b0, b1, b2, b3], axis=1)
                nms_outputs[:, :4] = origin_boxes

            final_outputs.append(nms_outputs)
        final_outputs = np.array(final_outputs)

        return final_outputs


if __name__ == "__main__":
    image_shape = (512, 512, 3)
    anchors = (
        np.array(
            [
                [162, 20],
                [95, 45],
                [54, 80],
                [183, 52],
                [172, 84],
                [214, 77],
                [189, 91],
                [198, 97],
                [208, 103],
                [190, 114],
                [224, 112],
                [248, 128],
            ]
        )
        / image_shape[0]
    )
    anchor_masks = np.array(
        [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]], dtype=np.int8
    )
    anchors = np.array(anchors, dtype=np.float32)
    yolo = Yolo(
        num_class=3,
        anchors=anchors,
        anchor_masks=anchor_masks,
        image_shape=image_shape,
        strides=[4, 8, 16, 32],
        use_net="large",
    )
    yolo.yolo_firi_modified.summary(line_length=200)
    #
    # from tensorflow.python.ops import summary_ops_v2
    # from tensorflow.python.keras.backend import get_graph
    #
    # tb_writer = tf.summary.create_file_writer('./logs')
    # with tb_writer.as_default():
    #     if not yolo3.yolo_model.run_eagerly:
    #         summary_ops_v2.graph(get_graph(), step=0)
