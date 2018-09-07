import csv
import cv2
import collections
import os

class_colors = {
    "Foundation":(0,0,255),
    "Incomplete":(0,165,255),
    "Complete":(0,255,0)
}

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Sample the bounding boxes csv file and rerun inference on the samples with lower threshold.')
    parser.add_argument('csv_file', type=str, help='the location of the csv file')
    args = parser.parse_args()

    filedir = os.path.dirname(args.csv_file)
    outputdir = os.path.join(filedir, "bounding")

    if not os.path.exists(outputdir):
        os.makedirs(outputdir)

    bounds_in_im = collections.defaultdict(list)

    with open(args.csv_file, "r") as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            impath, tlx, tly, brx, bry, classification = row

            if classification:
                bounds_in_im[impath].append(((int(tlx), int(tly)), (int(brx), int(bry)), classification))

    for impath in bounds_in_im:
        im = cv2.imread(impath)

        for classifications in bounds_in_im[impath]:
            minp, maxp, classification = classifications

            cv2.rectangle(im, (minp[0], minp[1]), (maxp[0], maxp[1]), class_colors[classification], 2)

        filename = os.path.basename(impath).replace(".tif", ".jpg")
        outfile = os.path.join(outputdir, filename)

        print(outfile)
        cv2.imwrite(outfile, im)