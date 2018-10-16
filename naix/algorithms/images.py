from skimage import measure
import numpy as np
from skimage.morphology import disk
from skimage.filters import threshold_otsu, rank
from skimage.util import img_as_ubyte
import cv2


def bin_via_local_otsu(image, radius=100):
    selem = disk(radius)
    local_otsu = rank.otsu(image, selem)
    return np.array(image >= local_otsu).astype(np.uint8)


def match_template(tpl, target, method=cv2.TM_SQDIFF_NORMED):
    # calculate the match region
    w, h = tpl.shape[::-1]

    res = cv2.matchTemplate(target, tpl, method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc

    bottom_right = (top_left[0] + w, top_left[1] + h)

    # calcuate similarity
    (xstart, ystart), (xend, yend) = top_left, bottom_right
    rmse = np.sqrt(measure.compare_mse(target[ystart:yend, xstart:xend], tpl))
    ssim = measure.compare_ssim(target[ystart:yend, xstart:xend], tpl)
    return top_left, bottom_right, rmse, ssim


def match_otsu_template(tpl, target):
    tg1= bin_via_local_otsu(target, radius=500)
    tg2 = bin_via_local_otsu(np.invert(target), radius=500)

    res1 = match_template(tpl, tg1)
    res2 = match_template(tpl, tg2)
    if res2[-1] > res1[-1]:
        return res2
    else:
        return res1


