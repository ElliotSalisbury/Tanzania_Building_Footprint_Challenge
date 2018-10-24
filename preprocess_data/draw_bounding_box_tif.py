import csv
import cv2
import collections
import os
from osgeo import gdal
import numpy as np

from preprocess_data.csv_processing import get_bounds_in_image


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
        ds = gdal.Open(impath)

        rows, cols, depth = ds.RasterYSize, ds.RasterXSize, ds.RasterCount

        for i, classifications in enumerate(bounds_in_im[impath]):
            minp, maxp, classification, score = classifications
            w = maxp[0] - minp[0]
            h = maxp[1] - minp[1]

            bounds_im = ds.ReadAsArray(int(minp[0]), int(minp[1]), w, h)
            bounds_im = np.moveaxis(bounds_im, 0, -1)
            bounds_im = cv2.cvtColor(bounds_im, cv2.COLOR_RGB2BGR)

            filename = os.path.basename(impath).replace(".tif", "_{}.jpg".format(i))
            outfile = os.path.join(outputdir, filename)

            cv2.imwrite(outfile, bounds_im)