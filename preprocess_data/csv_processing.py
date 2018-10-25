import csv
import collections

def get_bounds_in_image(csv_file):
    bounds_in_im = collections.defaultdict(list)

    with open(csv_file, "r") as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if not row:
                continue

            if len(row) == 6:
                impath, tlx, tly, brx, bry, classification = row
                scores = []
            else:
                impath, tlx, tly, brx, bry, classification, *scores = row

            if classification:
                bounds_in_im[impath].append(((int(tlx), int(tly)), (int(brx), int(bry)), classification, scores))

    return bounds_in_im