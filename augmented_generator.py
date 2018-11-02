import albumentations as A
from tif_generator import TiffGenerator
import numpy as np

class AugmentedGenerator(TiffGenerator):
    def __init__(
            self,
            csv_data_file,
            csv_class_file,
            base_dir=None,
            **kwargs
    ):
        super(AugmentedGenerator, self).__init__(csv_data_file, csv_class_file, base_dir, **kwargs)

    def preprocess_group_entry(self, image, annotations):
        # augment images
        image, annotations = self.augment_image(image, annotations)

        # preprocess the image
        image = self.preprocess_image(image)

        # randomly transform image and annotations
        # image, annotations = self.random_transform_group_entry(image, annotations)

        # resize image
        # image, image_scale = self.resize_image(image)

        # apply resizing to annotations too
        # annotations['bboxes'] *= image_scale

        return image, annotations

    def augment_image(self, image, annotations):
        new_annotations = annotations

        if annotations['bboxes'].any():
            annotation = {'image': image, 'bboxes': annotations['bboxes'], 'category_id': annotations['labels']}

            aug = self.get_aug([
                A.VerticalFlip(),
                A.HorizontalFlip(),
                A.RGBShift(),
                A.Blur(blur_limit=7),
                A.GaussNoise(),
                A.OpticalDistortion(distort_limit=0.2),
                A.GridDistortion(),
                A.ShiftScaleRotate(p=0.75, shift_limit=0.1, rotate_limit=45, scale_limit=0.2),
            ], min_area=(1024*0.05)**2)

            augmented = aug(**annotation)

            image = augmented['image']

            annotations['bboxes'] = np.array(augmented['bboxes'])
            annotations['labels'] = np.array(augmented['category_id'])

        return image, annotations

    def get_aug(self, aug, min_area=0., min_visibility=0.):
        return A.Compose(aug, bbox_params={'format': 'pascal_voc', 'min_area': min_area, 'min_visibility': min_visibility,
                                         'label_fields': ['category_id']})
