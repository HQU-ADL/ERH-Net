# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

import mmengine.fileio as fileio

from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class EndoVis2018(BaseSegDataset):
    """Pascal VOC dataset.

    Args:
        split (str): Split txt file for Pascal VOC.
    """
    METAINFO = dict(
        classes=('background-tissue','instrument-shaft','instrument-clasper','instrument-wrist','kidney-parenchyma','covered-kidney','thread','clamps','suturing-needle','suction-instrument','small-intestine','ultrasound-probe'),
        palette=[[0, 0, 0], [0, 255, 0], [0, 255, 255], [125, 255, 12],
                           [255, 55,0], [24, 55, 125], [187, 155, 25], [0, 255, 125],
                           [255, 255, 125], [123, 15, 175], [124, 155, 5], [12, 255, 141]])

    def __init__(self,
                 ann_file,
                 img_suffix='.jpg',
                 seg_map_suffix='.png',
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            ann_file=ann_file,
            **kwargs)
        assert fileio.exists(self.data_prefix['img_path'],
                             self.backend_args) and osp.isfile(self.ann_file)
