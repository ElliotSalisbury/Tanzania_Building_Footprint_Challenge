import argparse
import subprocess
import geojson
import os
import glob
from shapely.geometry import shape, Polygon
from osgeo import gdal
from osgeo import osr
import csv

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess the data into sizeable tiles.')
    parser.add_argument('in_dir', type=str, help='the location of the tiles')
    parser.add_argument('out_dir', type=str, help='the location of where we store the output')
    parser.add_argument('--window_size', type=int, default=1024, help='the size of the output tiles')
    parser.add_argument('--step_size', type=int, default=256, help='the size of the steps between output tiles')

    args = parser.parse_args()

    # call the script to make the tifs smaller
    subprocess.call(['./make_tiffs_smaller.sh', args.in_dir, args.out_dir, str(args.window_size), str(args.step_size)])

    print("======= PROCESSING SMALLER TILES' BOUNDING BOXES =======")
    output_csv_rows_all = []
    for geojson_file in glob.glob(os.path.join(args.in_dir, "*.geojson")):
        filename = os.path.basename(geojson_file).replace(".geojson", "")

        print(filename)

        with open(geojson_file) as f:
            data = geojson.load(f)

        if not data:
            continue

        shapeid_to_class = {}
        shapeid_to_shape = {}
        shapes = []
        for feature in data['features']:
            s = shape(feature['geometry'])
            id = feature['properties']['id']
            condition = feature['properties']['condition']

            shapeid_to_shape[id] = s
            shapeid_to_class[id] = condition
        geojson_cs = osr.SpatialReference()
        geojson_cs.SetFromUserInput(data['crs']['properties']['name'])


        # for each smaller tif in the output file that matches our filename
        for small_tif_file in glob.glob(os.path.join(args.out_dir, "tif", "{}*.tif".format(filename))):
            print("\t\t{}".format(small_tif_file))
            rect, raster_dims = get_tif_bounding_rect_in_srs(small_tif_file, geojson_cs)

            shapeids_in_tif = []
            for shapeid in shapeid_to_shape:
                shape = shapeid_to_shape[shapeid]
                intersection = rect.intersection(shape)
                if intersection.area > 0:
                    shapeids_in_tif.append(shapeid)

            output_csv_rows = []
            for shapeid in shapeids_in_tif:
                shape = shapeid_to_shape[shapeid]

                w,h = raster_dims
                t_minx, t_miny, t_maxx, t_maxy = rect.bounds
                s_minx, s_miny, s_maxx, s_maxy = shape.bounds

                p_minx = ((s_minx - t_minx) / (t_maxx - t_minx)) * w
                p_maxx = ((s_maxx - t_minx) / (t_maxx - t_minx)) * w
                p_miny = ((s_miny - t_miny) / (t_maxy - t_miny)) * h
                p_maxy = ((s_maxy - t_miny) / (t_maxy - t_miny)) * h

                p_minx = int(max(p_minx, 0))
                p_miny = int(max(p_miny, 0))
                p_maxx = int(min(p_maxx, w-1))
                p_maxy = int(min(p_maxy, h-1))

                classification = shapeid_to_class[shapeid]

                row = (small_tif_file, p_minx, p_miny, p_maxx, p_maxy, classification)

                output_csv_rows.append(row)

            if not output_csv_rows:
                output_csv_rows.append((small_tif_file,"","","","",""))

            output_csv_rows_all.extend(output_csv_rows)

    with open(os.path.join(args.out_dir,"bounds.csv"), "w") as f:
        writer = csv.writer(f)
        writer.writerows(output_csv_rows_all)


