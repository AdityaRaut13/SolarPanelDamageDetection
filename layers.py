import numpy as np
import math
import tensorflow as tf
from tensorflow import keras
from keras.layers import (
    Layer,
    Conv2D,
    BatchNormalization,
    ReLU,
    GlobalAveragePooling2D,
    Dense,
    SeparableConv2D,
    MaxPooling2D,
    Reshape,
    Conv1D,
    Activation,
    LeakyReLU,
    Concatenate,
)


class Conv(Layer):
    def __init__(
        self,
        out_channels,
        kernel_size=1,
        stride=1,
        padding="same",
        groups=1,
        act=True,
        activation=ReLU,
    ):
        super(Conv, self).__init__()
        self.act = act
        self.conv = Conv2D(
            filters=out_channels,
            groups=groups,
            kernel_size=kernel_size,
            padding=padding,
            use_bias=False,
            strides=stride,
        )
        self.bn = BatchNormalization()
        self.activation = activation()

    def call(self, inputs, *args, **kwargs):
        x = self.conv(inputs)
        x = self.bn(x)
        x = self.activation(x) if self.act else x
        return x


class SKLayerModified(Layer):
    def __init__(self, out_channel):
        super(SKLayerModified, self).__init__()
        self.conv1 = SeparableConv2D(
            filters=out_channel, kernel_size=(3, 3), padding="same"
        )
        self.bn1 = BatchNormalization()
        self.conv2 = SeparableConv2D(
            filters=out_channel,
            kernel_size=(3, 3),
            dilation_rate=(2, 2),
            padding="same",
        )
        self.bn2 = BatchNormalization()
        self.fc = Dense(out_channel, use_bias=False)

    def call(self, inputs, *args, **kwargs):
        x1 = self.bn1(self.conv1(inputs))
        x2 = self.bn2(self.conv2(inputs))
        fgp = GlobalAveragePooling2D()(x1 + x2)
        fcw = self.fc(fgp)
        fcw = Reshape([1, 1, fcw.shape[-1]])(fcw)
        return x1 * fcw + x2 * fcw + inputs


class CSP(Layer):
    def __init__(self, out_channels, kernel_size=3, stride=2):
        super().__init__()
        half_channels = out_channels // 2
        self.conv1 = Conv(out_channels=half_channels)
        self.conv2 = Conv(out_channels=half_channels)
        self.conv3 = Conv(out_channels=half_channels)
        self.conv4 = Conv(out_channels=half_channels)
        self.skLayer = SKLayerModified(out_channel=half_channels)
        self.conv5 = Conv(out_channels=half_channels)
        self.conv2d = Conv(out_channels=out_channels, kernel_size=kernel_size, stride=stride)

    def call(self, inputs):
        rs_conv1 = self.conv1(inputs)
        rs_conv2 = self.conv2(rs_conv1)
        rs_b1 = self.conv3(rs_conv1)
        rs_conv5 = self.conv5(self.skLayer(self.conv4(rs_conv2)))
        rs_b2 = rs_conv2 + rs_conv5
        rs = tf.concat([rs_b1, rs_b2], -1)
        return self.conv2d(rs)


class SPPF(Layer):
    def __init__(self, in_channels, out_channels, kernel_size=5):
        super(SPPF, self).__init__()
        in_half_channels = in_channels // 2
        self.conv1 = Conv(out_channels=in_half_channels)
        self.conv2 = Conv(out_channels=out_channels)
        self.maxpool = MaxPooling2D(
            pool_size=(kernel_size, kernel_size), strides=1, padding="same"
        )

    def call(self, inputs, *args, **kwargs):
        x = self.conv1(inputs)
        y1 = self.maxpool(x)
        y2 = self.maxpool(y1)
        y3 = self.maxpool(y2)
        concat_all = tf.keras.layers.Concatenate()([x, y1, y2, y3])
        output = self.conv2(concat_all)
        return output


class ECA_Layer(Layer):
    def __init__(self, in_channels, out_channels, gamma=5, b=1):
        super(ECA_Layer, self).__init__()
        self.out_channels = out_channels
        t = int(abs((math.log(in_channels, 2) + b) / gamma))
        k = t if t % 2 else t + 1
        self.conv = Conv1D(1, kernel_size=k, padding="same")
        self.act = Activation("sigmoid")

    def call(self, inputs_tensor, *args, **kwargs):
        x_global_avg_pool = GlobalAveragePooling2D()(inputs_tensor)
        x = Reshape((self.out_channels, 1))(x_global_avg_pool)
        x = self.conv(x)
        x = self.act(x)  # shape=[batch,chnnels,1]
        x = Reshape((1, 1, self.out_channels))(x)
        output = tf.multiply(inputs_tensor, x)
        return output


def eca_layer(inputs_tensor=None, gamma=2, b=1):
    """
    ECA-NET
    :param inputs_tensor: input_tensor.shape=[batchsize,h,w,channels]
    :param num:
    :param gamma:
    :param b:
    :return:
    """
    channels = keras.backend.int_shape(inputs_tensor)[-1]
    t = int(abs((math.log(channels, 2) + b) / gamma))
    k = t if t % 2 else t + 1
    x_global_avg_pool = GlobalAveragePooling2D()(inputs_tensor)
    x = Reshape((channels, 1))(x_global_avg_pool)
    x = Conv1D(1, kernel_size=k, padding="same", trainable=False)(x)
    x = Activation("sigmoid")(x)  # shape=[batch,chnnels,1]
    x = Reshape((1, 1, channels))(x)
    output = tf.multiply(inputs_tensor, x)
    return output


class AttentionConv(Layer):
    def __init__(
        self,
        out_channels,
        trainable=True,
        name=None,
        dtype=None,
        dynamic=False,
        **kwargs
    ):
        super().__init__(trainable, name, dtype, dynamic, **kwargs)
        self.out_channels = out_channels
        self.conv = Conv(out_channels=out_channels, activation=LeakyReLU)

    def call(self, inputs, *args, **kwargs):
        attopt = eca_layer(inputs)
        f1 = self.conv(attopt)
        return f1


class YoloHead(Layer):
    def __init__(
        self, image_shape, num_class, is_training, strides, anchors, anchors_masks
    ):
        super(YoloHead, self).__init__()
        self.image_shape = image_shape
        self.num_class = num_class
        self.is_training = is_training
        self.strides = strides
        self.anchors = anchors
        self.anchors_masks = anchors_masks
        self.grid = []
        self.anchor_grid = []
        for i, stride in enumerate(strides):
            grid, anchor_grid = self._make_grid(
                self.image_shape[0] // stride, self.image_shape[1] // stride, i
            )
            self.grid.append(grid)
            self.anchor_grid.append(anchor_grid)

    def call(self, inputs, *args, **kwargs):
        detect_res = []
        for i, pred in enumerate(inputs):
            if not self.is_training:
                pred = tf.sigmoid(pred)
                f_shape = pred.get_shape()
                # if len(self.grid) < self.anchor_masks.shape[0]:
                #     grid, anchor_grid = self._make_grid(f_shape[1], f_shape[2], i)
                #     self.grid.append(grid)
                #     self.anchor_grid.append(anchor_grid)
                # 这里把输出的值域从[0,1]调整到[0, image_shape]
                # pred_xy = (tf.sigmoid(pred[..., 0:2]) * 2. - 0.5 + self.grid[i]) * self.strides[i]
                pred_xy = (pred[..., 0:2] * 2.0 - 0.5 + self.grid[i]) * self.strides[i]
                # pred_wh = (tf.sigmoid(pred[..., 2:4]) * 2) ** 2 * self.anchor_grid[i]
                pred_wh = (
                    (pred[..., 2:4] * 2) * (pred[..., 2:4] * 2) * self.anchor_grid[i]
                )
                # print(self.grid)
                pred_obj = pred[..., 4:5]
                # pred_cls = tf.keras.layers.Softmax()(pred[..., 5:])
                pred_cls = pred[..., 5:]
                cur_layer_pred_res = Concatenate(axis=-1)(
                    [pred_xy, pred_wh, pred_obj, pred_cls]
                )

                # cur_layer_pred_res = tf.reshape(cur_layer_pred_res, [self.batch_size, -1, self.num_class + 5])
                cur_layer_pred_res = Reshape(
                    [f_shape[1] * f_shape[2] * f_shape[3], self.num_class + 5]
                )(cur_layer_pred_res)
                detect_res.append(cur_layer_pred_res)
            else:
                detect_res.append(pred)
        return detect_res if self.is_training else tf.concat(detect_res, axis=1)

    def _make_grid(self, h, w, i):
        cur_layer_anchors = self.anchors[self.anchors_masks[i]] * np.array(
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


def nms(
    image_shape, predicts, conf_thres=0.45, iou_thres=0.2, max_det=300, max_nms=3000
):
    """原yolov5简化版nms, 不用multi label, 不做merge box
    :param image_shape:
    :param predicts:
    :param conf_thres:
    :param iou_thres:
    :param max_det:
    :return: [batch, nms_nums, (x1, y1, x2, y2, conf, cls)]
    """
    output = []

    # 这里遍历每个batch,也就是每张图, 输出的3层预测已经做了合并处理成[batch, -1, 5+num_class]
    for i, predict in enumerate(predicts):
        # predict = predict.numpy()
        # 首先只要那些目标概率大于阈值的
        obj_mask = predict[..., 4] > conf_thres
        predict = predict[obj_mask]

        # 没有满足的数据则跳过去下一张
        if not predict.shape[0]:
            continue

        # 类别概率乘上了目标概率, 作为最终判别概率
        # print(np.max(predict[:, 5:]), np.min(predict[:, 5:]))
        # print(np.max(predict[:, 4:5]), np.min(predict[:, 4:5]))
        predict[:, 5:] *= predict[:, 4:5]

        x1 = np.maximum(predict[:, 0] - predict[:, 2] / 2, 0)
        y1 = np.maximum(predict[:, 1] - predict[:, 3] / 2, 0)
        x2 = np.minimum(predict[:, 0] + predict[:, 2] / 2, image_shape[1])
        y2 = np.minimum(predict[:, 1] + predict[:, 3] / 2, image_shape[0])
        box = np.concatenate(
            [x1[:, None], y1[:, None], x2[:, None], y2[:, None]], axis=-1
        )
        # Detections matrix [n, (x1, y1, x2, y2, conf, cls)]
        max_cls_ids = np.array(predict[:, 5:].argmax(axis=1), dtype=np.float32)
        max_cls_score = predict[:, 5:].max(axis=1)
        predict = np.concatenate(
            [box, max_cls_score[:, None], max_cls_ids[:, None]], axis=1
        )[np.reshape(max_cls_score > 0.1, (-1,))]

        n = predict.shape[0]
        if not n:
            continue
        elif n > max_nms:
            predict = predict[predict[:, 4].argsort()[::-1][:max_nms]]

        # 为每个类别乘上一个大数,box再加上这个偏移, 做nms时就可以在类内做
        cls = predict[:, 5:6] * 4096
        # 边框加偏移
        boxes, scores = predict[:, :4] + cls, predict[:, 4]
        nms_ids = tf.image.non_max_suppression(
            boxes=boxes, scores=scores, max_output_size=max_det, iou_threshold=iou_thres
        )

        output.append(predict[nms_ids.numpy()])

    return output
