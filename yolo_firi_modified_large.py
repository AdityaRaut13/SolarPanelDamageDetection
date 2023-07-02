import tensorflow as tf
from layers import Conv, CSP, SPPF, SKLayerModified
from attention import cbam_block
from keras.layers import UpSampling2D, Reshape
from keras.models import Model


class Yolo_FIRI_Modified_Large:
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
        f5 = SKLayerModified(1024)(SPPF(in_channels=1024, out_channels=1024)(f4))

        # neck
        nf1 = CSP(out_channels=1024, kernel_size=1, stride=1)(f5)
        nf1 = cbam_block(nf1)
        nf1_up = UpSampling2D((2, 2), interpolation="bilinear")(nf1)

        nf2 = CSP(out_channels=512, kernel_size=1, stride=1)(
            tf.concat([nf1_up, f3], axis=-1)
        )
        nf2 = cbam_block(nf2)
        nf2_up = UpSampling2D((2, 2), interpolation="bilinear")(nf2)

        nf3 = CSP(out_channels=256, kernel_size=1, stride=1)(
            tf.concat([nf2_up, f2], axis=-1)
        )
        nf3 = cbam_block(nf3)
        nf3_up = UpSampling2D((2, 2), interpolation="bilinear")(nf3)

        nf4 = Conv(out_channels=128, kernel_size=1, stride=1)(
            tf.concat([nf3_up, f1], axis=-1)
        )
        nf4 = cbam_block(nf4)

        op4 = CSP(out_channels=res_channels, kernel_size=1, stride=1)(nf1)
        op3 = CSP(out_channels=res_channels, kernel_size=1, stride=1)(nf2)
        op2 = CSP(out_channels=res_channels, kernel_size=1, stride=1)(nf3)
        op1 = CSP(out_channels=res_channels, kernel_size=1, stride=1)(nf4)

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
    yolo_firi_modified = Yolo_FIRI_Modified_Large(
        image_shape=(512, 512, 3), batch_size=2, num_class=3, anchors_per_location=3
    )
    model = yolo_firi_modified.build_graph()
    model.summary(line_length=300)
