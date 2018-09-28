import albumentations as A
from keras_retinanet.preprocessing.csv_generator import CSVGenerator


class AugmentedGenerator(CSVGenerator):
    def __init__(
            self,
            csv_data_file,
            csv_class_file,
            base_dir=None,
            **kwargs
    ):
        super(AugmentedGenerator, self).__init__(csv_data_file, csv_class_file, base_dir, **kwargs)

    def preprocess_group_entry(self, image, annotations):
        # preprocess the image
        image = self.preprocess_image(image)

        # augment images
        image, annotations = self.augment_image(image, annotations)

        # randomly transform image and annotations
        image, annotations = self.random_transform_group_entry(image, annotations)

        # resize image
        image, image_scale = self.resize_image(image)

        # apply resizing to annotations too
        annotations[:, :4] *= image_scale

        return image, annotations

    def augment_image(self, image, annotations):
        annotation = {'image': image, 'bboxes': annotations[:, :4], 'category_id': annotations[:, 4]}

        aug = self.get_aug([
            A.VerticalFlip(),
            A.HorizontalFlip(),
            A.RGBShift(),
            A.Blur(),
            A.GaussNoise(),
        ])
        augmented = aug(**annotation)

        image = augmented['image']

        annotations[:, :4] = augmented['bboxes']
        annotations[:, 4] = augmented['category_id']

        return image, annotations

    def get_aug(self, aug, min_area=0., min_visibility=0.):
        return A.Compose(aug, bbox_params={'format': 'pascal_voc', 'min_area': min_area, 'min_visibility': min_visibility,
                                         'label_fields': ['category_id']})
