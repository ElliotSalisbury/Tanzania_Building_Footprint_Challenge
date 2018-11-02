"""
Copyright 2017-2018 yhenon (https://github.com/yhenon/)
Copyright 2017-2018 Fizyr (https://fizyr.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import random

import cv2
import numpy as np
from keras_retinanet.preprocessing.csv_generator import CSVGenerator
from osgeo import gdal
from shapely.geometry import Polygon


def _clamp(n, minn, maxn):
    return max(min(maxn, n), minn)


def _order(a, b):
    if a > b:
        return b, a
    return a, b


class TiffGenerator(CSVGenerator):
    """ Generate data for a custom CSV dataset.

    See https://github.com/fizyr/keras-retinanet#csv-datasets for more information.
    """

    def __init__(
            self,
            csv_data_file,
            csv_class_file,
            base_dir=None,
            **kwargs
    ):
        kwargs['group_method'] = 'random'
        super().__init__(csv_data_file, csv_class_file, base_dir, **kwargs)

        self.get_image_data_flat()

    def get_image_data_flat(self):
        self.image_data_flat = []
        self.image_data_to_image_index = []
        for image_index, image_name in enumerate(self.image_data):
            self.image_data_flat.extend(self.image_data[image_name])
            self.image_data_to_image_index.extend([image_index, ] * len(self.image_data[image_name]))

        self.annotation_index_window = {}

    def size(self):
        """ Size of the dataset.
        """
        self.get_image_data_flat()
        return len(self.image_data_flat)

    def load_image(self, annotation_index):
        """ Load an image at the annotation_index.
        """
        image_index = self.image_data_to_image_index[annotation_index]

        ds = gdal.Open(self.image_path(image_index))
        tif_height, tif_width = ds.RasterYSize, ds.RasterXSize

        annotation = self.image_data_flat[annotation_index]

        # randomly select a window around this annotation
        border = 50
        annotation['x1'], annotation['x2'] = _order(annotation['x1'], annotation['x2'])
        annotation['y1'], annotation['y2'] = _order(annotation['y1'], annotation['y2'])

        min_x = (annotation['x2'] + border) - self.image_min_side
        min_y = (annotation['y2'] + border) - self.image_min_side
        max_x = annotation['x1'] - border
        max_y = annotation['y1'] - border

        min_x, max_x = _order(max_x, min_x)
        min_y, max_y = _order(max_y, min_y)

        window_x = random.randint(min_x, max_x)
        window_y = random.randint(min_y, max_y)

        # correct for being out the range of the tif
        window_x = max(min(window_x, tif_width - self.image_min_side), 0)
        window_y = max(min(window_y, tif_height - self.image_min_side), 0)

        ds_array = ds.ReadAsArray(window_x, window_y, self.image_min_side, self.image_min_side)
        image = np.moveaxis(ds_array, 0, -1)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # store the window coordinates so that we can get the annotations later
        self.annotation_index_window[annotation_index] = (window_x, window_y)

        return image

    def load_annotations(self, annotation_index):
        """ Load annotations for an image_index.
        """
        image_index = self.image_data_to_image_index[annotation_index]

        path = self.image_names[image_index]
        annotations = {'labels': np.empty((0,)), 'bboxes': np.empty((0, 4))}

        window = self.annotation_index_window[annotation_index]
        del self.annotation_index_window[annotation_index]

        windowpoly = Polygon([
            window,
            (window[0], window[1] + self.image_min_side),
            (window[0] + self.image_min_side, window[1] + self.image_min_side),
            (window[0] + self.image_min_side, window[1])
        ])

        for idx, annot in enumerate(self.image_data[path]):
            bbox = Polygon([
                (float(annot['x1']), float(annot['y1'])),
                (float(annot['x1']), float(annot['y2'])),
                (float(annot['x2']), float(annot['y2'])),
                (float(annot['x2']), float(annot['y1']))
            ])
            intersection = windowpoly.intersection(bbox)
            if intersection.area > 0:
                x1 = _clamp(float(annot['x1']) - window[0], 0, self.image_min_side - 1)
                y1 = _clamp(float(annot['y1']) - window[1], 0, self.image_min_side - 1)
                x2 = _clamp(float(annot['x2']) - window[0], 0, self.image_min_side - 1)
                y2 = _clamp(float(annot['y2']) - window[1], 0, self.image_min_side - 1)

                if x1 != x2 and y1 != y2:
                    annotations['labels'] = np.concatenate((annotations['labels'], [self.name_to_label(annot['class'])]))
                    annotations['bboxes'] = np.concatenate((annotations['bboxes'], [[x1,y1,x2,y2]]))

        return annotations
