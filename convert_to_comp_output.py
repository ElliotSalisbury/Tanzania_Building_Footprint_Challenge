import csv
import cv2
import collections
import os
from osgeo import gdal
import numpy as np
from osgeo import ogr, osr
from shapely.geometry import Point, Polygon


from preprocess_data.csv_processing import get_bounds_in_image

def rect_2_corners(minp,maxp):
    return ((minp[0], minp[1]),
            (minp[0], maxp[1]),
            (maxp[0], maxp[1]),
            (maxp[0], minp[1]),
            (minp[0], minp[1]),
            )

def pixel_2_geo(ds, pixels):
    xoffset, px_w, rot1, yoffset, rot2, px_h = ds.GetGeoTransform()
    old_cs = osr.SpatialReference()
    old_cs.ImportFromWkt(ds.GetProjectionRef())

    # create the new coordinate system
    new_cs = osr.SpatialReference()
    new_cs.ImportFromEPSG(4326)

    # create a transform object to convert between coordinate systems
    transform = osr.CoordinateTransformation(old_cs, new_cs)

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

        # get the coordinates in lat long
        latlong = transform.TransformPoint(posX, posY)

        geo.append(latlong)

    return geo

def points_2_poly(points):
    poly = Polygon(points)
    poly = ogr.CreateGeometryFromWkb(poly.wkb)
    return poly

def main(csv_file, output_dir):
    bounds_in_im = get_bounds_in_image(csv_file)

    for impath in bounds_in_im:
        ds = gdal.Open(impath)

        rows, cols, depth = ds.RasterYSize, ds.RasterXSize, ds.RasterCount

        filename = os.path.basename(impath).split(".")[0]
        outfile = os.path.join(output_dir, filename + ".csv")

        rows = []
        for i, classifications in enumerate(bounds_in_im[impath]):
            minp, maxp, classification, scores = classifications

            building_id = len(rows) + 1
            conf_foundation = 0.01  # scores[0]
            conf_unfinished = 0.01  # scores[1]
            conf_completed = 0.98  # scores[2]

            corners_pixel = rect_2_corners(minp, maxp)
            corners_geo = pixel_2_geo(ds, corners_pixel)

            poly_pixel = points_2_poly(corners_pixel)
            poly_geo = points_2_poly(corners_geo)

            coords_pixel = poly_pixel.ExportToWkt()
            coords_geo = poly_geo.ExportToWkt()

            row = [building_id, conf_foundation, conf_unfinished, conf_completed, coords_geo, coords_pixel]
            rows.append(row)

        with open(outfile, "w+") as f:
            writer = csv.writer(f, lineterminator='\n')
            writer.writerow(["building_id","conf_foundation","conf_unfinished","conf_completed","coords_geo","coords_pixel"])
            writer.writerows(rows)

if __name__ == '__main__':
    # import argparse
    #
    # parser = argparse.ArgumentParser(description='Sample the bounding boxes csv file and rerun inference on the samples with lower threshold.')
    # parser.add_argument('csv_file', type=str, help='the location of the csv file')
    # parser.add_argument('output_dir', type=str, help='the location of the output directory')
    # args = parser.parse_args()


    # main(args.csv_file, args.output_dir)

    tiles = ["034", "047", "067", "074", "076", "135", "181"]
    t_2_h = ["5ae242fd0b093000130afd33",
             "5b00370f2b6a08001185f129",
             "5ae318220b093000130afd99",
             "5b00370f2b6a08001185f12b",
             "5ae36dd70b093000130afdbb",
             "5ae38a540b093000130aff24",
             "5ae38a540b093000130afed0"]

    for tile in tiles:
        arg = r"C:\Users\Elliot\Desktop\openai\TEST\processed\grid_{}.csv".format(tile)

        main(arg, r"C:\Users\Elliot\Desktop\openai\TEST\processed\comp")


    for i, tile in enumerate(tiles):
        hash = t_2_h[i]

        in_name = r"C:\Users\Elliot\Desktop\openai\TEST\processed\comp\grid_{}.csv".format(tile)
        out_name = in_name.replace("grid_{}".format(tile), hash)
        os.rename(in_name, out_name)