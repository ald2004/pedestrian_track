import cv2
import numpy as np
import time
import pint_mini

class FeatureExtractionPintAPIAlgorithm():
    def __init__(self, config):
        self.config = config
        self._build()

    def _build(self):
        in_shape = [1, 128, 64, 3]
        options = pint_mini.Options()
        options.fusion_on = self.config["fusion_on"]
        options.pitch_on = self.config["pitch_on"]
        in_names = ['images']
        out_names = ['truediv']

        if self.config["tuning"]:
            options.fusion_on = self.config["fusion_on"]
            options.pitch_on = self.config["pitch_on"]
            options.prt_mode = self.config["prt_mode"]

        if self.config['quantization_enable']:
            options.json_path = self.config["tuning_i8"]
            graph = pint_mini.buildGraph(self.config['model_path_pmd'], 3)
            self.sess = pint_mini.createSessionFromPmd(graph, options=options)
        else:
            options.json_path = self.config["tuning_fp32"]
            #print(self.config['model_path'])
            graph = pint_mini.buildGraph(self.config['model_path'], 0, in_names=in_names, out_names=out_names)
            self.sess = pint_mini.createSession(graph, [in_shape], options)

    def _post_process(self, det):
        return det

    def run(self, images, boxess):
        imgs = self._pre_process(images)
        feature = []
        # start = time.time()
        for img in imgs:
            # f = []
            for boxes in boxess:
                for box in boxes:
                    img_roi = self.extract_image_patch(img, box, (self.config['width'], self.config['height']))
                    im = img_roi[np.newaxis, :]
                    result = self.sess.predict(im)
                    ret = self._post_process(result)
                    # f.append(ret[0])
                    feature.append(ret[0])
                    # print(ret[0].shape)
            # feature.append(f)
            # feature.extend(f)

        # end = time.time()
        # print("#" * 40)
        # print("reid run time is {:.2f} ms".format((end - start) * 1000))

        return feature


    def _pre_process(self, imgs):
        def read_image(input_image, height, width):
            if len(input_image.shape) == 2:  # if the image is gray, convert it to bgr format
                print("Warning:gray image", input_image)
                input_image = cv2.cvtColor(input_image, cv2.COLOR_GRAY2BGR)
            img = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
            return img

        tmp = []
        for img in imgs:
            img_ = read_image(img, self.config['height'], self.config['width'])
            tmp.append(img_)

        return tmp

    def extract_image_patch(self, img, bbox, patch_shape):
        bbox = np.array(bbox)
        bbox[2:] += bbox[:2]
        bbox = bbox.astype(np.int)

        bbox[:2] = np.maximum(0, bbox[:2])
        bbox[2:] = np.minimum(np.asarray(img.shape[:2][::-1]), bbox[2:])
        if np.any(bbox[:2] > bbox[2:]):
            return None

        sx, sy, ex, ey = bbox
        image = img[sy:ey, sx:ex]
        image = cv2.resize(image, patch_shape)

        return image

model = FeatureExtractionPintAPIAlgorithm