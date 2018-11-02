import csv
import glob
import os
import time

# import miscellaneous modules
import cv2
import keras
import numpy as np


# set tf backend to allow memory to grow, instead of claiming everything
import tensorflow as tf
# import keras_retinanet
from keras_retinanet import models
from keras_retinanet.utils.image import preprocess_image, resize_image

from image_utils import large_tiff_to_windows, non_max_suppression_fast
from preprocess_data.draw_bounding_box import visualize_bbox

def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)


def get_boxes(img, model):
    img = img.copy()
    preprocess_image(img)
    img, scale = resize_image(img, min_side=1024, max_side=1024)

    # process image
    start = time.time()
    boxes, scores, labels = model.predict_on_batch(np.expand_dims(img, axis=0))
    print("processing time: ", time.time() - start)

    # correct for image scale
    boxes /= scale

    return boxes, scores, labels

def load_classes_csv(csv_path):
    class_dict = {}
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        for i, class_row in enumerate(reader):
            class_dict[int(class_row[1])] = class_row[0]

        return class_dict

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Run inference on the given tiles.')
    parser.add_argument('in_dir', type=str, help='the location of the tiles')
    parser.add_argument('model_path', type=str, help='the location of the model weights')
    parser.add_argument('classes_csv', type=str, help='the location of the classes.csv')
    parser.add_argument('--threshold', type=float, default=0.2,
                        help='the threshold of the confidence in the boxes we output')
    parser.add_argument('--nms_overlap', type=float, default=0.15,
                        help='the overlap threshold we require before discarding overlapping boxes')
    args = parser.parse_args()

    # set the modified tf session as backend in keras
    keras.backend.tensorflow_backend.set_session(get_session())

    # load retinanet model
    model = models.load_model(args.model_path, backbone_name='resnet50')

    # load label to names mapping for visualization purposes
    labels_to_names = load_classes_csv(args.classes_csv)

    # get the images we want to run on
    if os.path.isdir(args.in_dir):
        filepaths = glob.glob(os.path.join(args.in_dir, "*.tif"))
    else:
        filepaths = []

        with open(args.in_dir, "r") as f:
            reader = csv.reader(f)
            for i, row in enumerate(reader):
                impath, tlx, tly, brx, bry, classification = row

                filepaths.append(impath)
    # filepaths = ["./preprocess_data/test.jpg", ]

    # run inference on each image
    for f_i, filepath in enumerate(filepaths):
        print("{} {}/{}".format(filepath, f_i, len(filepaths)))

        outdir = os.path.join(os.path.dirname(filepath), "processed")
        filename = os.path.basename(filepath).split(".")[0]

        outcsvpath = os.path.join(outdir, filename + ".csv")
        # if os.path.exists(outcsvpath):
        #     continue

        allrows = []
        windows = large_tiff_to_windows(filepath, window_step=256)
        # windows = [[cv2.imread(filepath),(0,0)],]
        for w_i, window_tuple in enumerate(windows):
            window = window_tuple[0]
            tl = window_tuple[1]
            total = window_tuple[2]

            # ignore completely blank tiles
            if window.min() == window.max():
                continue

            boxes, scores, labels = get_boxes(window, model)

            for box, score, label in zip(boxes[0], scores[0], labels[0]):
                if score < args.threshold:
                    break

                b = np.array(box).astype(int)

                visualize_bbox(window, (b[0], b[1], b[2], b[3]), label, score=score)

                tl_bounds = tl + b[:2]
                br_bounds = tl + b[2:]

                row = [filepath, tl_bounds[0], tl_bounds[1], br_bounds[0], br_bounds[1], labels_to_names[label], score]

                allrows.append(row)

                with open(os.path.join(outdir, str(filename) + "_intermediate.csv"), "a") as f:
                    writer = csv.writer(f)
                    writer.writerow(row)

            # cv2.imshow("results", window)
            # cv2.waitKey(1)
            print("{} {} t:{}/{}  w:{}/{}".format(filepath, tl, f_i + 1, len(filepaths), w_i, total))

            # cv2.imshow("window", window)
            # cv2.waitKey(1)

        # Non Max Supress the boxes so we remove the overlapped runs
        if allrows:
            nprows = np.array(allrows)
            boxes = nprows[:, [1,2,3,4,6,0,5]]
            boxes = non_max_suppression_fast(boxes, args.nms_overlap)
            boxes = boxes[:, [5, 0, 1, 2, 3, 6, 4]]

        # write the final output to file
        with open(outcsvpath, "w") as f:
            writer = csv.writer(f)
            writer.writerows(boxes)
