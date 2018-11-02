import csv
import cv2
import collections
import os
from osgeo import gdal
import numpy as np

from csv_processing import get_bounds_in_image

def rect_2_windowcoords(window, minp, maxp):
    new_min = (int(minp[0] - window[0]), int(minp[1] - window[1]))
    new_max = (int(maxp[0] - window[0]), int(maxp[1] - window[1]))

    return new_min, new_max

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Sample the bounding boxes csv file and rerun inference on the samples with lower threshold.')
    parser.add_argument('csv_file', type=str, help='the location of the csv file')
    parser.add_argument('--window_size', type=int, default=1024, help='the size of the window around the bounding box')
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
            w2 = (maxp[0] - minp[0]) / 2
            h2 = (maxp[1] - minp[1])
            c = (minp[0] + w2, minp[1] + h2)

            rect_p = (max(c[0] - args.window_size/2, 0), max(c[1] - args.window_size/2, 0))

            w = int(min(args.window_size, cols - rect_p[0]))
            h = int(min(args.window_size, rows - rect_p[1]))

            bounds_im = ds.ReadAsArray(int(rect_p[0]), int(rect_p[1]), w, h)
            bounds_im = np.moveaxis(bounds_im, 0, -1)
            bounds_im = cv2.cvtColor(bounds_im, cv2.COLOR_RGB2BGR)

            # get rects in this image
            im_minp, im_maxp = rect_2_windowcoords(rect_p, minp, maxp)
            rects = [(im_minp, im_maxp),]

            for j, bounds in enumerate(bounds_in_im[impath]):
                if i == j:
                    continue
                minp, maxp, classification, score = bounds
                im_minp, im_maxp = rect_2_windowcoords(rect_p, minp, maxp)

                if (im_minp[0] >= 0 and im_minp[0] < args.window_size) or (im_minp[1] >= 0 and im_minp[1] < args.window_size) or (im_maxp[0] >= 0 and im_maxp[0] < args.window_size) or (im_maxp[1] >= 0 and im_maxp[1] < args.window_size):
                    rects.append((im_minp, im_maxp))

            for j, rect in enumerate(rects):
                color = (75,56,56)
                if j == 0:
                    color = (0, 255, 24)

                cv2.rectangle(bounds_im, rect[0], rect[1], color, thickness=3)

            filename = os.path.basename(impath).replace(".tif", "_{}.jpg".format(i))
            outfile = os.path.join(outputdir, filename)

            cv2.imwrite(outfile, bounds_im)