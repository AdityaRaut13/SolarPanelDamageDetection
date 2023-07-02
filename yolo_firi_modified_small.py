import tensorflow as tf
from layers import Conv, CSP, SPPF, SKLayerModified, AttentionConv
from keras.layers import UpSampling2D, Reshape
from keras.models import Model


class Yolo_FIRI_Modified:
    def __init__(self, image_shape, batch_size, num_class, anchors_per_location):
        self.image_shape = image_shape
        self.batch_size = batch_size
        self.num_class = num_class
        self.anchors_per_location = anchors_per_location

    def build_graph(self):
        """
        :param inputs:
        :return: p7: [batch, h/8, w/8, anchors, num_class+5]
                 p8: [batch, h/16, w/16, anchors, num_class+5]
                 p9: [batch, h/32, w/32, anchors, num_class+5]
        """
        res_channels = (self.num_class + 5) * self.anchors_per_location
        inputs = tf.keras.Input(shape=self.image_shape, batch_size=self.batch_size)

        # focus layer
        x = Conv(out_channels=64, kernel_size=6, stride=2, padding="same")(inputs)

        # backbone

        f1 = CSP(out_channels=128)(x)
        f2 = CSP(out_channels=256)(f1)
        f3 = CSP(out_channels=512)(f2)
        f4 = CSP(out_channels=1024)(f3)
        op4 = SKLayerModified(512)(SPPF(in_channels=1024, out_channels=512)(f4))

        # neck

        # 1/32
        op4 = AttentionConv(out_channels=res_channels)(op4)
        ir3 = UpSampling2D((2, 2), interpolation="bilinear")(op4)

        # 1/16
        op3 = AttentionConv(out_channels=res_channels)(tf.concat([ir3, f3], -1))
        ir2 = UpSampling2D((2, 2), interpolation="bilinear")(op3)

        # 1/8
        op2 = AttentionConv(out_channels=res_channels)(tf.concat([ir2, f2], -1))
        ir1 = UpSampling2D((2, 2), interpolation="bilinear")(op2)

        # 1/4
        op1 = AttentionConv(out_channels=res_channels)(tf.concat([ir1, f1], -1))

        # reshaping the output
        # From [ batch , H, W, C]  to [batch,H,W,anchors, 5 + num_classes]

        op4 = Reshape(
            [
                self.image_shape[0] // 32,
                self.image_shape[1] // 32,
                self.anchors_per_location,
                self.num_class + 5,
            ]
        )(op4)

        op3 = Reshape(
            [
                self.image_shape[0] // 16,
                self.image_shape[1] // 16,
                self.anchors_per_location,
                self.num_class + 5,
            ]
        )(op3)

        op2 = Reshape(
            [
                self.image_shape[0] // 8,
                self.image_shape[1] // 8,
                self.anchors_per_location,
                self.num_class + 5,
            ]
        )(op2)

        op1 = Reshape(
            [
                self.image_shape[0] // 4,
                self.image_shape[1] // 4,
                self.anchors_per_location,
                self.num_class + 5,
            ]
        )(op1)

        model = Model(inputs=inputs, outputs=[op1, op2, op3, op4])
        return model


if __name__ == "__main__":
    yolo_firi_modified = Yolo_FIRI_Modified(
        image_shape=(512, 512, 3), batch_size=2, num_class=3, anchors_per_location=3
    )
    model = yolo_firi_modified.build_graph()
    model.summary(line_length=300)
