import numpy as np
from osgeo import gdal
import cv2

def im2windows(im, window_size=225, window_step=225):
    rows,cols,depth = im.shape
    row_start_index = np.arange(0, rows-window_size, step=window_step)
    col_start_index = np.arange(0, cols-window_size, step=window_step)

    num_windows = row_start_index.shape[0] * col_start_index.shape[0]
    windows = []
    windows_tl_indexs = np.zeros((num_windows, 2), dtype=np.int)
    i = 0
    for row in row_start_index:
        for col in col_start_index:
            windows.append(im[row:row+window_size, col:col+window_size, :])
            windows_tl_indexs[i] = [col, row]
            i = i + 1

    return windows, windows_tl_indexs

def large_tiff_to_windows(filepath, window_size=1024, window_step=512):
    ds = gdal.Open(filepath)

    rows, cols, depth = ds.RasterYSize, ds.RasterXSize, ds.RasterCount
    row_start_index = np.arange(0, rows - window_size, step=window_step)
    col_start_index = np.arange(0, cols - window_size, step=window_step)

    for row in row_start_index:
        for col in col_start_index:
            ds_array = ds.ReadAsArray(int(col), int(row), window_size, window_size)
            satim = np.moveaxis(ds_array, 0, -1)
            satim = cv2.cvtColor(satim, cv2.COLOR_RGB2BGR)

            yield satim, [col, row]


def non_max_suppression_fast(boxes, overlapThresh):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    score = boxes[:, 4]

    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(score)

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the first index in the indexes list and add the
        # index value to the list of picked indexes
        i = idxs[0]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[1:]])
        yy1 = np.maximum(y1[i], y1[idxs[1:]])
        xx2 = np.minimum(x2[i], x2[idxs[1:]])
        yy2 = np.minimum(y2[i], y2[idxs[1:]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[1:]]

        # delete all indexes from the index list that have
        idxs = np.delete(idxs[1:], np.where(overlap > overlapThresh)[0])

    # return only the bounding boxes that were picked using the
    # integer data type
    return boxes[pick]