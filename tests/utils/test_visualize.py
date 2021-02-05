# Copyright (c) Facebook, Inc. and its affiliates.

import os.path
import unittest

import numpy as np
from mmf.utils.features.visualizing_image import SingleImageViz, get_data


OBJ_URL = (
    "https://raw.githubusercontent.com/airsplay/py-bottom-up-attention/master/"
    "demo/data/genome/1600-400-20/objects_vocab.txt"
)
ATTR_URL = (
    "https://raw.githubusercontent.com/airsplay/py-bottom-up-attention/master/"
    "demo/data/genome/1600-400-20/attributes_vocab.txt"
)


class TestVisualize(unittest.TestCase):
    def test_single_image_viz(self):

        objids = get_data(OBJ_URL)
        attrids = get_data(ATTR_URL)

        cwd = os.path.abspath(os.path.dirname(__file__))
        img_path = os.path.join(cwd, "./features/img.npy")

        img = np.load(img_path)

        info_path = img_path = os.path.join(cwd, "./features/img_info.npy")

        output_dict = np.load(info_path, allow_pickle=True).item()

        frcnn_visualizer = SingleImageViz(img, id2obj=objids, id2attr=attrids)
        frcnn_visualizer.draw_boxes(
            output_dict.get("boxes"),
            output_dict.pop("obj_ids"),
            output_dict.pop("obj_probs"),
            output_dict.pop("attr_ids"),
            output_dict.pop("attr_probs"),
        )

        buffer = frcnn_visualizer._get_buffer()

        buffer_path = os.path.join(cwd, "./features/img_buffer.npy")

        buffer_valid = np.load(buffer_path)

        self.assertTrue((buffer == buffer_valid).all())
