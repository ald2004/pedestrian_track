import numpy as np


class YOLO_Predict(object):
    def __init__(self,  img_size=[416, 416], class_num=80):
        self.anchors = [[10, 13], [16, 30], [33, 23],
                         [30, 61], [62, 45], [59,  119],
                         [116, 90], [156, 198], [373,326]]
        self.img_size = img_size
        self.class_num = class_num

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def reorg_layer(self, feature_map, anchors):
        '''
        feature_map: a feature_map from [feature_map_1, feature_map_2, feature_map_3] returned
            from `forward` function
        anchors: shape: [3, 2]
        '''
        # NOTE: size in [h, w] format! don't get messed up!

        grid_size = list(feature_map.shape)[1:3]
        # the downscale ratio in height and weight
        ratio = self.img_size[0] / grid_size[0], self.img_size[1] / grid_size[1]
        # rescale the anchors to the feature_map
        # NOTE: the anchor is in [w, h] format!
        rescaled_anchors = [(anchor[0] / ratio[1], anchor[1] / ratio[0]) for anchor in anchors]

        feature_map = np.reshape(feature_map, [-1, grid_size[0], grid_size[1], 3, 5 + self.class_num])

        # split the feature_map along the last dimension
        # shape info: take 416x416 input image and the 13*13 feature_map for example:
        # box_centers: [N, 13, 13, 3, 2] last_dimension: [center_x, center_y]
        # box_sizes: [N, 13, 13, 3, 2] last_dimension: [width, height]
        # conf_logits: [N, 13, 13, 3, 1]
        # prob_logits: [N, 13, 13, 3, class_num]
        temp = np.split(feature_map, [2, 4, 5], axis=-1)  # np.split is different from tf.split
        box_centers, box_sizes, conf_logits, prob_logits = temp[0], temp[1], temp[2], temp[3]
        box_centers = self.sigmoid(box_centers)

        # np.concatenate()

        # use some broadcast tricks to get the mesh coordinates
        grid_x = np.arange(grid_size[1], dtype=np.int32)
        grid_y = np.arange(grid_size[0], dtype=np.int32)
        grid_x, grid_y = np.meshgrid(grid_x, grid_y)
        x_offset = np.reshape(grid_x, (-1, 1))
        y_offset = np.reshape(grid_y, (-1, 1))
        x_y_offset = np.concatenate([x_offset, y_offset], axis=-1)
        # shape: [13, 13, 1, 2]
        x_y_offset = np.reshape(x_y_offset, [grid_size[0], grid_size[1], 1, 2]).astype(np.float32)

        # get the absolute box coordinates on the feature_map
        box_centers = box_centers + x_y_offset
        # rescale to the original image scale
        box_centers = box_centers * ratio[::-1]

        # avoid getting possible nan value with tf.clip_by_value
        box_sizes = np.exp(box_sizes) * rescaled_anchors
        # rescale to the original image scale
        box_sizes = box_sizes * ratio[::-1]

        # shape: [N, 13, 13, 3, 4]
        # last dimension: (center_x, center_y, w, h)
        boxes = np.concatenate([box_centers, box_sizes], axis=-1)

        # shape:
        # x_y_offset: [13, 13, 1, 2]
        # boxes: [N, 13, 13, 3, 4], rescaled to the original image scale
        # conf_logits: [N, 13, 13, 3, 1]
        # prob_logits: [N, 13, 13, 3, class_num]
        return x_y_offset, boxes, conf_logits, prob_logits


    def predict(self, feature_maps):
        '''
        Receive the returned feature_maps from `forward` function,
        the produce the output predictions at the test stage.
        '''
        feature_map_1, feature_map_2, feature_map_3 = feature_maps

        feature_map_anchors = [(feature_map_1, self.anchors[6:9]),
                               (feature_map_2, self.anchors[3:6]),
                               (feature_map_3, self.anchors[0:3])]
        reorg_results = [self.reorg_layer(feature_map, anchors) for (feature_map, anchors) in feature_map_anchors]

        def _reshape(result):
            x_y_offset, boxes, conf_logits, prob_logits = result
            grid_size = list(x_y_offset.shape)[:2]
            boxes = np.reshape(boxes, [-1, grid_size[0] * grid_size[1] * 3, 4])
            conf_logits = np.reshape(conf_logits, [-1, grid_size[0] * grid_size[1] * 3, 1])
            prob_logits = np.reshape(prob_logits, [-1, grid_size[0] * grid_size[1] * 3, self.class_num])
            # shape: (take 416*416 input image and feature_map_1 for example)
            # boxes: [N, 13*13*3, 4]
            # conf_logits: [N, 13*13*3, 1]
            # prob_logits: [N, 13*13*3, class_num]
            return boxes, conf_logits, prob_logits

        boxes_list, confs_list, probs_list = [], [], []
        for result in reorg_results:
            boxes, conf_logits, prob_logits = _reshape(result)
            confs = self.sigmoid(conf_logits)
            probs = self.sigmoid(prob_logits)
            boxes_list.append(boxes)
            confs_list.append(confs)
            probs_list.append(probs)

        boxes = np.concatenate(boxes_list, axis=1)
        confs = np.concatenate(confs_list, axis=1)
        probs = np.concatenate(probs_list, axis=1)

        temp = np.split(boxes, [1, 2, 3], axis=-1)
        center_x, center_y, width, height = temp[0], temp[1], temp[2], temp[3]
        x_min = center_x - width / 2
        y_min = center_y - height / 2
        x_max = center_x + width / 2
        y_max = center_y + height / 2

        boxes = np.concatenate([x_min, y_min, x_max, y_max], axis=-1)

        return boxes, confs, probs
