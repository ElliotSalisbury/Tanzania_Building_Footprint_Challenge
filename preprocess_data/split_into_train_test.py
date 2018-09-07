import csv
import collections
import os
import random

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Split the bounds.csv into a training and validation set.')
    parser.add_argument('csv_file', type=str, help='the location of the csv file')
    parser.add_argument('-validation_percent', type=float, default=0.2, help='The percentage of images we use in the validation set')
    args = parser.parse_args()

    filedir = os.path.dirname(args.csv_file)
    outputdir = os.path.join(filedir)

    # we only want images with bounding boxes in our validation set
    bounds_in_im = collections.defaultdict(list)
    all_im_rows = collections.defaultdict(list)
    with open(args.csv_file, "r") as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            impath, tlx, tly, brx, bry, classification = row

            all_im_rows[impath].append(row)

            if classification:
                bounds_in_im[impath].append(((int(tlx), int(tly)), (int(brx), int(bry)), classification))

    validation_count = int(len(bounds_in_im) * args.validation_percent)
    validation_images = random.sample(bounds_in_im.keys(), validation_count)
    train_images = set(all_im_rows.keys()).difference(validation_images)

    validation_rows = []
    for im in validation_images:
        validation_rows.extend(all_im_rows[im])

    train_rows = []
    for im in train_images:
        train_rows.extend(all_im_rows[im])

    with open(os.path.join(outputdir, "bounds_train.csv"), "w") as f:
        writer = csv.writer(f)
        writer.writerows(train_rows)

    with open(os.path.join(outputdir, "bounds_val.csv"), "w") as f:
        writer = csv.writer(f)
        writer.writerows(validation_rows)