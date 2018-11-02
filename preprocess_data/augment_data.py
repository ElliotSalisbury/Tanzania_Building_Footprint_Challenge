import argparse
import collections
import os
import csv

import cv2
import albumentations as A

from draw_bounding_box import visualize_bbox

BOX_COLOR = (255, 0, 0)
TEXT_COLOR = (255, 255, 255)


def read_csv_file(csv_file_path):
    filedir = os.path.dirname(csv_file_path)
    outputdir = os.path.join(filedir)

    # we only want images with bounding boxes in our validation set
    bounds_in_im = collections.defaultdict(list)
    all_im_rows = collections.defaultdict(list)
    with open(csv_file_path, "r") as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            impath, tlx, tly, brx, bry, classification = row

            all_im_rows[impath].append(row)

            if classification:
                bounds_in_im[impath].append(((int(tlx), int(tly)), (int(brx), int(bry)), classification))

    return all_im_rows, bounds_in_im


def visualize(annotation):
    img = annotation['image'].copy()

    for idx, bbox in enumerate(annotation['bboxes']):
        category = annotation['category_id'][idx]

        img = visualize_bbox(img, bbox, category)

    return img



def get_aug(aug, min_area=0., min_visibility=0.3):
    return A.Compose(aug, bbox_params={'format': 'pascal_voc', 'min_area': min_area, 'min_visibility': min_visibility,
                                     'label_fields': ['category_id']})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='augment the preprocessed data into more variants.')
    parser.add_argument('train_csv', type=str, help='the location of the training csv')
    parser.add_argument('out_dir', type=str, help='the location of where we store the augmented output')
    args = parser.parse_args()

    all_im_rows, bounds_in_im = read_csv_file(args.train_csv)

    for im_path in bounds_in_im:
        orig = cv2.imread(im_path)

        bounds = []
        categories = []
        for bbox in bounds_in_im[im_path]:
            box = bbox[0], bbox[1]

            bounds.append((box[0][0], box[0][1], box[1][0], box[1][1]))
            categories.append(bbox[2])

        # construct data struct for the image
        annotations = {'image': orig, 'bboxes': bounds, 'category_id': categories}


        img = visualize(annotations)


        for i in range(100):
            aug = get_aug([
                A.VerticalFlip(),
                A.HorizontalFlip(),
                A.RGBShift(),
                A.Blur(blur_limit=9),
                A.GaussNoise(),
                A.OpticalDistortion(distort_limit=0.2),
                A.ShiftScaleRotate(shift_limit=0.1, rotate_limit=45, scale_limit=0.2),
                A.GridDistortion()
            ], min_area=(1024 * 0.05) ** 2)

            augmented = aug(**annotations)
            aug = visualize(augmented)

            cv2.imshow("aug", aug)
            cv2.imshow("img", img)
            print("done")
            cv2.waitKey(-1)