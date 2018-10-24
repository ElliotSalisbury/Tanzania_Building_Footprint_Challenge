import csv
import cv2
import collections
import os
from preprocess_data.csv_processing import get_bounds_in_image

class_colors = {
    "Foundation":(0,0,255),
    "Incomplete":(0,165,255),
    "Complete":(0,255,0)
}

def visualize_bbox(img, bbox, class_name, color=(255,255,255), thickness=2, score=None):
    x_min, y_min, x_max, y_max = [int(e) for e in bbox]

    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)

    if score is None:
        label_text = class_name
    else:
        label_text = "{}: {}".format(class_name, score)

    ((text_width, text_height), _) = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
    cv2.rectangle(img, (x_min, y_min), (x_min + text_width, y_min + int(1.3 * text_height)), color, -1)
    cv2.putText(img, label_text, (x_min, y_min + text_height), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255,255,255),
                lineType=cv2.LINE_AA)
    return img

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Sample the bounding boxes csv file and rerun inference on the samples with lower threshold.')
    parser.add_argument('csv_file', type=str, help='the location of the csv file')
    args = parser.parse_args()

    filedir = os.path.dirname(args.csv_file)
    outputdir = os.path.join(filedir, "bounding")

    if not os.path.exists(outputdir):
        os.makedirs(outputdir)

    bounds_in_im = get_bounds_in_image(args.csv_file)

    for impath in bounds_in_im:
        im = cv2.imread(impath)

        for classifications in bounds_in_im[impath]:
            minp, maxp, classification, score = classifications

            visualize_bbox(im, (minp[0], minp[1], maxp[0], maxp[1]), classification, class_colors[classification], score=score)

        filename = os.path.basename(impath).replace(".tif", ".jpg")
        outfile = os.path.join(outputdir, filename)

        print(outfile)
        cv2.imwrite(outfile, im)