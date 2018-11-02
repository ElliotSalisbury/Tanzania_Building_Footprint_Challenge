import argparse
import subprocess
import geojson
import os
import glob
from shapely.geometry import shape, Polygon
from osgeo import gdal
from osgeo import osr
import csv
import collections
import random

def get_tif_bounding_rect_in_srs(tif_file, cs):
    # open the tif file
    ds = gdal.Open(tif_file)

    # read its bounds in the tif's definied coordinate system
    width = ds.RasterXSize
    height = ds.RasterYSize
    gt = ds.GetGeoTransform()

    minx = gt[0]
    miny = gt[3] + width * gt[4] + height * gt[5]
    maxx = gt[0] + width * gt[1] + height * gt[2]
    maxy = gt[3]

    # get the tifs coordinate system, and create a transformation to the given coordinate system
    old_cs = osr.SpatialReference()
    old_cs.ImportFromWkt(ds.GetProjectionRef())
    transform = osr.CoordinateTransformation(old_cs, cs)

    # transform our min and max bounds to the new coordinate system
    minlatlng = transform.TransformPoint(minx, miny)
    maxlatlng = transform.TransformPoint(maxx, maxy)

    # make the bounding shapely rect
    rect = Polygon([minlatlng[:2], (minlatlng[0], maxlatlng[1]), maxlatlng[:2], (maxlatlng[0], minlatlng[1])])

    return rect, (width,height)

def split_into_train_set(csv_file, validation_percent, window_size):
    filedir = os.path.dirname(csv_file)
    outputdir = os.path.join(filedir)

    # we only want images with bounding boxes in our validation set
    bounds_in_im = collections.defaultdict(list)
    ims_per_tile = collections.defaultdict(set)
    all_im_rows = collections.defaultdict(list)
    with open(csv_file, "r") as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            impath, tlx, tly, brx, bry, classification = row

            all_im_rows[impath].append(row)

            if classification:
                bounds_in_im[impath].append(((int(tlx), int(tly)), (int(brx), int(bry)), classification))

                filename = os.path.basename(impath)
                filename = os.path.splitext(filename)[0]
                *filenames, W, H = filename.split("_")
                filename = "_".join(filenames)
                ims_per_tile[filename].add(impath)

    validation_count = int(len(bounds_in_im) * validation_percent)
    validation_images = random.sample(bounds_in_im.keys(), validation_count)
    exclude_set = set(validation_images)

    validation_rows = []
    for i, im in enumerate(validation_images):
        print("\t{}/{} {}".format(i, len(validation_images), im))
        validation_rows.extend(all_im_rows[im])

        #remove anything from the train images that might overlapm the validation images
        filename = os.path.basename(im)
        filename = os.path.splitext(filename)[0]
        *filenames, W, H = filename.split("_")
        filename = "_".join(filenames)
        W = int(W)
        H = int(H)

        for t_im in ims_per_tile[filename].copy():
            t_filename = os.path.basename(t_im)
            t_filename = os.path.splitext(t_filename)[0]
            *t_filenames, t_W, t_H = t_filename.split("_")

            t_W = int(t_W)
            t_H = int(t_H)

            if abs(W-t_W) < window_size and abs(H-t_H) < window_size:
                ims_per_tile[filename].remove(t_im)
                exclude_set.add(t_im)


                print("\t\texcluding: {}".format(t_im))

    train_images = set(all_im_rows.keys()).difference(exclude_set)
    train_rows = []
    for im in train_images:
        train_rows.extend(all_im_rows[im])

    with open(os.path.join(outputdir, "bounds_train.csv"), "w") as f:
        writer = csv.writer(f)
        writer.writerows(train_rows)

    with open(os.path.join(outputdir, "bounds_val.csv"), "w") as f:
        writer = csv.writer(f)
        writer.writerows(validation_rows)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess the data into sizeable tiles.')
    parser.add_argument('in_dir', type=str, help='the location of the tiles')
    parser.add_argument('out_dir', type=str, help='the location of where we store the output')
    parser.add_argument('-validation_percent', type=float, default=0.005,
                        help='The percentage of images we use in the validation set')

    args = parser.parse_args()
    bounds_path = os.path.join(args.out_dir, "bounds.csv")
    if not os.path.exists(bounds_path):
        output_csv_rows_all = []
        for geojson_file in glob.glob(os.path.join(args.in_dir, "*.geojson")):
            filename = os.path.basename(geojson_file).replace(".geojson", "")
            tifpath = os.path.join(args.in_dir, "{}.tif".format(filename))

            print(filename)

            with open(geojson_file) as f:
                data = geojson.load(f)

            if not data:
                continue

            geojson_cs = osr.SpatialReference()
            geojson_cs.SetFromUserInput(str(data['crs']['properties']['name']))
            rect, raster_dims = get_tif_bounding_rect_in_srs(tifpath, geojson_cs)

            for id, feature in enumerate(data['features']):
                if 'geometry' in feature and 'coordinates' in feature['geometry'] and len(feature['geometry']['coordinates']) > 0:
                    s = Polygon(feature['geometry']['coordinates'][0])
                    s = Polygon.from_bounds(*s.bounds)
                    condition = feature['properties']['condition']

                    if condition:
                        w, h = raster_dims
                        t_minx, t_miny, t_maxx, t_maxy = rect.bounds
                        s_minx, s_miny, s_maxx, s_maxy = s.bounds

                        p_minx = ((s_minx - t_minx) / (t_maxx - t_minx)) * w
                        p_maxx = ((s_maxx - t_minx) / (t_maxx - t_minx)) * w
                        p_maxy = (1 - ((s_miny - t_miny) / (t_maxy - t_miny))) * h
                        p_miny = (1 - ((s_maxy - t_miny) / (t_maxy - t_miny))) * h

                        p_minx = int(max(p_minx, 0))
                        p_miny = int(max(p_miny, 0))
                        p_maxx = int(min(p_maxx, w - 1))
                        p_maxy = int(min(p_maxy, h - 1))

                        if p_minx < p_maxx and p_miny < p_maxy:
                            row = (tifpath, p_minx, p_miny, p_maxx, p_maxy, condition)

                            output_csv_rows_all.append(row)

        with open(bounds_path, "w") as f:
            writer = csv.writer(f, lineterminator='\n')
            writer.writerows(output_csv_rows_all)

    # split the data into train and validation
    # print("======= VALIDATION / TRAINING SETS =======")
    # split_into_train_set(bounds_path, args.validation_percent, args.window_size)


