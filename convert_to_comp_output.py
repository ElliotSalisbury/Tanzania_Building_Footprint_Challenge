import csv
import cv2
import collections
import os
from osgeo import gdal
import numpy as np
from osgeo import ogr
from shapely.geometry import Point, Polygon


from preprocess_data.csv_processing import get_bounds_in_image

def rect_2_corners(minp,maxp):
    return ((minp[0], minp[1]),
            (minp[0], maxp[1]),
            (maxp[0], maxp[1]),
            (maxp[0], minp[1]),
            )

def pixel_2_geo(ds, pixels):
    xoffset, px_w, rot1, yoffset, rot2, px_h = ds.GetGeoTransform()

    geo = []
    for pixel in pixels:
        x,y = pixel
        # supposing x and y are your pixel coordinate this
        # is how to get the coordinate in space.
        posX = px_w * x + rot1 * y + xoffset
        posY = rot2 * x + px_h * y + yoffset

        # shift to the center of the pixel
        posX += px_w / 2.0
        posY += px_h / 2.0

        geo.append((posX, posY))

    return geo

def points_2_poly(points):
    poly = Polygon(points)
    poly = ogr.CreateGeometryFromWkb(poly.wkb)
    return poly


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Sample the bounding boxes csv file and rerun inference on the samples with lower threshold.')
    parser.add_argument('csv_file', type=str, help='the location of the csv file')
    parser.add_argument('output_dir', type=str, help='the location of the output directory')
    args = parser.parse_args()

    bounds_in_im = get_bounds_in_image(args.csv_file)


    for impath in bounds_in_im:
        ds = gdal.Open(impath)

        rows, cols, depth = ds.RasterYSize, ds.RasterXSize, ds.RasterCount

        filename = os.path.basename(impath).split(".")[0]
        outfile = os.path.join(args.output_dir, filename+ ".csv")

        rows = []
        for i, classifications in enumerate(bounds_in_im[impath]):
            minp, maxp, classification, scores = classifications

            building_id = len(rows) + 1
            conf_foundation = scores[0]
            conf_unfinished = scores[1]
            conf_completed = scores[2]

            corners_pixel = rect_2_corners(minp, maxp)
            corners_geo = pixel_2_geo(ds, corners_pixel)

            poly_pixel = points_2_poly(corners_pixel)
            poly_geo = points_2_poly(corners_geo)

            coords_pixel = poly_pixel.ExportToWkt()
            coords_geo = poly_geo.ExportToWkt()

            row = [building_id, conf_foundation, conf_unfinished, conf_completed, coords_geo, coords_pixel]
            rows.append(row)

        with open(outfile, "w+") as f:
            writer = csv.writer(f)
            writer.writerows(rows)