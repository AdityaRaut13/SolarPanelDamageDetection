import os
import cv2
import random
import re
import traceback
from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
from augmentation import Augmentation


class CoCoDataGenrator:
    def __init__(
        self,
        coco_annotation_file,
        train_img_nums=-1,
        img_shape=(640, 640, 3),
        batch_size=1,
        max_instances=50,
        using_argument=False,
        include_crowd=False,
        include_mask=False,
        include_keypoint=False,
        need_down_image=True,
        download_image_path=os.path.dirname(os.path.abspath(__file__))
        + "/"
        + "./coco_2017_val_images/",
    ):
        # 设置要训练的图片数, -1表示全部
        self.train_img_nums = train_img_nums
        # 是否需要下载图片数据, 只有官方CoCo数据才需要下载, 自己打标转CoCo格式不需要
        self.need_down_image = need_down_image
        # 设置下载保存coco json文件中图片的目录
        self.download_image_path = download_image_path
        # 图片最终resize+padding后的大小
        self.img_shape = img_shape
        self.batch_size = batch_size
        # 此参数为保证不同size的box,mask能padding到一个batch里
        self.max_instances = max_instances
        # 是否使用数据增强
        self.using_argument = using_argument
        # 是否输出包含crowd类型数据
        self.include_crowd = include_crowd
        # 是否输出包含mask分割数据
        self.include_mask = include_mask
        # 是否输出包含keypoint数据
        self.include_keypoint = include_keypoint
        self.coco_annotation_file = coco_annotation_file

        self.current_batch_index = 0
        self.total_batch_size = 0
        self.img_ids = []
        self.coco = COCO(annotation_file=coco_annotation_file)
        self.load_data()
        if self.need_down_image:
            self.download_image_files()
        self.augmentation = Augmentation()

    def load_data(self):
        # 初步过滤数据是否包含crowd
        target_img_ids = []
        for k in self.coco.imgToAnns:
            annos = self.coco.imgToAnns[k]
            if annos:
                annos = list(
                    filter(lambda x: x["iscrowd"] == self.include_crowd, annos)
                )
                if annos:
                    target_img_ids.append(k)

        if self.train_img_nums > 0:
            # np.random.shuffle(target_img_ids)
            target_img_ids = target_img_ids[: self.train_img_nums]

        self.total_batch_size = len(target_img_ids) // self.batch_size
        self.img_ids = target_img_ids

    def download_image_files(self):
        """下载coco图片数据"""
        if not os.path.exists(self.download_image_path):
            os.makedirs(self.download_image_path)

        if len(os.listdir(self.download_image_path)) > 0:
            print(
                "image files already downloaded! size: {}".format(
                    len(os.listdir(self.download_image_path))
                )
            )

        for i, img_id in enumerate(self.img_ids):
            file_path = self.download_image_path + "./{}.jpg".format(img_id)
            if os.path.isfile(file_path):
                print("already exist file: {}".format(file_path))
            else:
                if self.coco.imgs[img_id].get("coco_url"):
                    try:
                        im = io.imread(self.coco.imgs[img_id]["coco_url"])
                        io.imsave(file_path, im)
                        print(
                            "save image {}, {}/{}".format(
                                file_path, i + 1, len(self.img_ids)
                            )
                        )
                    except Exception as e:
                        traceback.print_exc()
                        print(
                            "current img_id: ", img_id, "current img_file: ", file_path
                        )

    def next_batch(self):
        if self.current_batch_index >= self.total_batch_size:
            self.current_batch_index = 0
            self._on_epoch_end()

        batch_img_ids = self.img_ids[
            self.current_batch_index
            * self.batch_size : (self.current_batch_index + 1)
            * self.batch_size
        ]
        batch_imgs = []
        batch_bboxes = []
        batch_labels = []
        batch_masks = []
        batch_keypoints = []
        valid_nums = []
        for img_id in batch_img_ids:
            # {"img":, "bboxes":, "labels":, "masks":, "key_points":}
            data = self._data_generation(image_id=img_id)
            if len(np.shape(data["imgs"])) > 0 and len(data["bboxes"]) > 0:
                batch_imgs.append(data["imgs"])
                batch_labels.append(data["labels"])
                batch_bboxes.append(data["bboxes"])
                valid_nums.append(data["valid_nums"])

                if self.include_mask:
                    batch_masks.append(data["masks"])

                if self.include_keypoint:
                    batch_keypoints.append(data["keypoints"])

        self.current_batch_index += 1

        if len(batch_imgs) < self.batch_size:
            return self.next_batch()

        output = {
            "imgs": np.array(batch_imgs, dtype=np.int32),
            "bboxes": np.array(batch_bboxes, dtype=np.int16),
            "labels": np.array(batch_labels, dtype=np.int8),
            "masks": np.array(batch_masks, dtype=np.int8),
            "keypoints": np.array(batch_keypoints, dtype=np.int16),
            "valid_nums": np.array(valid_nums, dtype=np.int8),
        }

        return output

    def _on_epoch_end(self):
        np.random.shuffle(self.img_ids)

    def _resize_im(self, origin_im, bboxes):
        """对图片/mask/box resize

        :param origin_im
        :param bboxes
        :return im_blob: [h, w, 3]
                gt_boxes: [N, [ymin, xmin, ymax, xmax]]
        """
        im_shape = np.shape(origin_im)
        im_size_max = np.max(im_shape[0:2])
        im_scale = float(self.img_shape[0]) / float(im_size_max)

        # resize原始图片
        im_resize = cv2.resize(
            origin_im,
            None,
            None,
            fx=im_scale,
            fy=im_scale,
            interpolation=cv2.INTER_LINEAR,
        )
        im_resize_shape = np.shape(im_resize)
        im_blob = np.zeros(self.img_shape, dtype=np.float32)
        im_blob[0 : im_resize_shape[0], 0 : im_resize_shape[1], :] = im_resize

        # resize对应边框
        bboxes_resize = np.array(bboxes * im_scale, dtype=np.int16)

        return im_blob, bboxes_resize

    def _resize_mask(self, origin_masks):
        """resize mask数据
        :param origin_mask:
        :return: mask_resize: [instance, h, w]
                 gt_boxes: [N, [ymin, xmin, ymax, xmax]]
        """
        mask_shape = np.shape(origin_masks)
        mask_size_max = np.max(mask_shape[0:3])
        im_scale = float(self.img_shape[0]) / float(mask_size_max)

        # resize mask/box
        gt_boxes = []
        masks_resize = []
        for m in origin_masks:
            m = np.array(m, dtype=np.float32)
            m_resize = cv2.resize(
                m, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR
            )
            m_resize = np.array(m_resize >= 0.5, dtype=np.int8)

            # 计算bdbox
            h, w = np.shape(m_resize)
            rows, cols = np.where(m_resize)
            # [xmin, ymin, xmax, ymax]
            xmin = np.min(cols) if np.min(cols) >= 0 else 0
            ymin = np.min(rows) if np.min(rows) >= 0 else 0
            xmax = np.max(cols) if np.max(cols) <= w else w
            ymax = np.max(rows) if np.max(rows) <= h else h
            bdbox = [xmin, ymin, xmax, ymax]
            gt_boxes.append(bdbox)

            mask_blob = np.zeros(
                (self.img_shape[0], self.img_shape[1], 1), dtype=np.float32
            )
            mask_blob[0:h, 0:w, 0] = m_resize
            masks_resize.append(mask_blob)

        # [instance_num, [xmin, ymin, xmax, ymax]]
        gt_boxes = np.array(gt_boxes, dtype=np.int16)
        # [h, w, instance_num]
        masks_resize = np.concatenate(masks_resize, axis=-1)

        return masks_resize, gt_boxes

    def _load_image(self, image_id):
        """读图片"""
        img_coco_url_file = str(self.coco.imgs[image_id].get("coco_url", ""))
        img_url_file = str(self.coco.imgs[image_id].get("url", ""))
        img_local_file = (
            str(self.coco.imgs[image_id].get("file_name", ""))
            .encode("unicode_escape")
            .decode()
        )
        img_local_file = os.path.join(
            os.path.dirname(self.coco_annotation_file), img_local_file
        )
        img_local_file = re.sub(r"\\\\", "/", img_local_file)

        img = []

        if os.path.isfile(img_local_file):
            img = io.imread(img_local_file)
        elif img_coco_url_file.startswith("http"):
            download_image_file = self.download_image_path + "./{}.jpg".format(image_id)
            if not os.path.isfile(download_image_file):
                print("download image from {}".format(img_coco_url_file))
                im = io.imread(img_coco_url_file)
                io.imsave(download_image_file, im)
                print("save image {}".format(download_image_file))
            img = io.imread(download_image_file)
        elif img_url_file.startswith("http"):
            download_image_file = self.download_image_path + "./{}.jpg".format(image_id)
            if not os.path.isfile(download_image_file):
                print("download image from {}".format(img_url_file))
                im = io.imread(img_url_file)
                io.imsave(download_image_file, im)
                print("save image {}".format(download_image_file))
            img = io.imread(download_image_file)
        else:
            return img

        if len(np.shape(img)) < 2:
            return img
        elif len(np.shape(img)) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            # img = np.expand_dims(img, axis=-1)
            # img = np.pad(img, [(0, 0), (0, 0), (0, 2)])
        else:
            img = img[:, :, ::-1]
        return img

    def _load_annotations(self, image_id):
        """读取打标数据"""
        anno_ids = self.coco.getAnnIds(imgIds=image_id, iscrowd=self.include_crowd)
        bboxes = []
        labels = []
        masks = []
        keypoints = []
        for i in anno_ids:
            # 边框, 处理成左上右下坐标
            ann = self.coco.anns[i]
            bbox = ann["bbox"]
            xmin, ymin, w, h = bbox
            xmin = int(xmin)
            ymin = int(ymin)
            xmax = int(xmin + w)
            ymax = int(ymin + h)
            bboxes.append([xmin, ymin, xmax, ymax])
            # 类别ID
            label = ann["category_id"]
            labels.append(label)
            # 实例分割
            if self.include_mask:
                # [instances, h, w]
                mask = self.coco.annToMask(ann)
                masks.append(mask)
            if self.include_keypoint and ann.get("keypoints"):
                keypoint = ann["keypoints"]
                # 处理成[x,y,v] 其中v=0表示没有此点,v=1表示被挡不可见,v=2表示可见
                keypoint = np.reshape(keypoint, [-1, 3])
                keypoints.append(keypoint)
        outputs = {
            "labels": np.array(labels),
            "bboxes": np.array(bboxes),
            "masks": np.array(masks),
            "keypoints": np.array(keypoints),
        }
        return outputs

    def _data_generation(self, image_id):
        """拉取coco标记数据, 目标边框, 类别, mask
        :param image_id:
        :return:
        """
        # 输出包含5个东西, 不需要则为空
        outputs = {
            "imgs": [],
            "labels": [],
            "bboxes": [],
            "masks": [],
            "keypoints": [],
            "valid_nums": 0,
        }

        valid_nums = 0
        if self.using_argument:
            # {"imgs":, "labels":, "bboxes":, "masks":, "keypoints":}
            argument_output = self._random_arguments(image_id)
            img = argument_output["imgs"]
            labels = argument_output["labels"]
            bboxes = argument_output["bboxes"]
            masks = argument_output["masks"]
            keypoints = argument_output["keypoints"]
        else:
            img = self._load_image(image_id)
            # {"labels":, "bboxes":, "masks":, "keypoints":}
            annotations = self._load_annotations(image_id)
            if not img.shape[0]:
                return outputs
            labels = annotations["labels"]
            bboxes = annotations["bboxes"]
            masks = annotations["masks"]
            keypoints = annotations["keypoints"]

        if len(labels) > self.max_instances:
            bboxes = bboxes[: self.max_instances, :]
            labels = labels[: self.max_instances]
            valid_nums = self.max_instances
            # batch_bboxes.append(data['bboxes'][:self.max_instances, :])
            # batch_labels.append(data['labels'][:self.max_instances])
            # valid_nums.append(self.max_instances)
        else:
            pad_num = self.max_instances - len(labels)
            bboxes = np.pad(bboxes, [(0, pad_num), (0, 0)])
            labels = np.pad(labels, [(0, pad_num)])
            valid_nums = self.max_instances - pad_num
            # batch_bboxes.append(np.pad(data['bboxes'], [(0, pad_num), (0, 0)]))
            # batch_labels.append(np.pad(data['labels'], [(0, pad_num)]))
            # valid_nums.append(len(data['labels']))
        # print(bboxes)
        labels = np.array(labels, dtype=np.int8)
        bboxes = np.array(bboxes, dtype=np.int16)
        img_resize, bboxes_resize = self._resize_im(origin_im=img, bboxes=bboxes)

        # 处理最终数据 mask
        if self.include_mask:
            # [h, w, instances]
            masks, _ = self._resize_mask(origin_masks=masks)
            if np.shape(masks)[2] > self.max_instances:
                masks = masks[: self.max_instances, :, :]
            else:
                pad_num = self.max_instances - np.shape(masks)[2]
                masks = np.pad(masks, [(0, 0), (0, 0), (0, pad_num)])

            outputs["masks"] = masks
            # outputs['bboxes'] = bboxes

        # 处理最终数据 keypoint
        if self.include_keypoint:
            keypoints = np.array(keypoints, dtype=np.int8)
            outputs["keypoints"] = keypoints

        outputs["imgs"] = img_resize
        outputs["labels"] = labels
        outputs["bboxes"] = bboxes_resize
        outputs["valid_nums"] = valid_nums

        return outputs

    def _random_arguments(self, image_id):
        """数据增强, 目前只做目标检测类的增强, 分割的不支持"""

        outputs = {
            "imgs": [],
            "labels": [],
            "bboxes": [],
            "masks": [],
            "keypoints": [],
        }

        img = self._load_image(image_id)
        # {"labels":, "bboxes":, "masks":, "keypoints":}
        annotations = self._load_annotations(image_id)
        outputs["imgs"] = img
        outputs["labels"] = annotations["labels"]
        outputs["bboxes"] = annotations["bboxes"]
        outputs["masks"] = annotations["masks"]
        outputs["keypoints"] = annotations["keypoints"]

        if img.shape[0]:
            r = random.random()
            if r < 0.5:
                input_images = [img]
                input_bboxes = [
                    np.concatenate(
                        [annotations["bboxes"], annotations["labels"][:, None]], axis=-1
                    )
                ]
                # 马赛克4图拼接
                ids = random.sample(self.img_ids, 3)
                for id in ids:
                    img_i = self._load_image(id)
                    if img_i.shape[0]:
                        input_images.append(img_i)
                        annotations_i = self._load_annotations(id)
                        input_bboxes.append(
                            np.concatenate(
                                [
                                    annotations_i["bboxes"],
                                    annotations_i["labels"][:, None],
                                ],
                                axis=-1,
                            )
                        )
                    else:
                        return outputs
                new_img, new_bboxes = self.augmentation.random_mosaic(
                    input_images, input_bboxes, self.img_shape[0]
                )
                if len(new_bboxes) > 0:
                    new_labels = new_bboxes[:, -1]
                    outputs["imgs"] = new_img
                    outputs["labels"] = new_labels
                    outputs["bboxes"] = new_bboxes[:, :-1]

            elif r < 0.75:
                # 平移/旋转/缩放/错切/透视变换
                input_bboxes = np.concatenate(
                    [annotations["bboxes"], annotations["labels"][:, None]], axis=-1
                )
                new_img, new_bboxes = self.augmentation.random_perspective(
                    img, input_bboxes
                )
                # 曝光, 饱和, 亮度调整
                new_img = self.augmentation.random_hsv(new_img)
                if len(new_bboxes) > 0:
                    new_labels = new_bboxes[:, -1]
                    outputs["imgs"] = new_img
                    outputs["labels"] = new_labels
                    outputs["bboxes"] = new_bboxes[:, :-1]
            # print(r, outputs['bboxes'])
        return outputs


if __name__ == "__main__":
    # dataset\valid\_annotations.coco.json
    val_data = CoCoDataGenrator(
        coco_annotation_file="../dataset/valid/_annotations.coco.json",
        need_down_image=False,
    )
    outputs = val_data.next_batch()
    print(outputs["labels"])
