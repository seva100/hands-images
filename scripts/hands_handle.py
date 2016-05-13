# -*- coding: utf-8 -*-
"""
Hands pictures handling.

@author: Artem Sevastopolsky, 2016
"""

from operator import attrgetter
from collections import deque
import numpy as np
import scipy as sp
import scipy.misc
import imhandle as imh
import skimage
import skimage.measure
import skimage.morphology
import cv2


def expand(img, ans, row, col, cluster, radius=3, tol=2, adjacency=8):
    '''Expands cluster from point based on the following criterion: 
    color of new pixel should be close (with tolerance tol) to mean color of growing component.
    adjacency parameter:
    adjacency=8 means search over fully-connected neighborhood,
    adjacency=4 search over row and column of specified radius 
    (particularly useful when radius=1)
    '''
    q = deque([(row, col)])
    compn_sum = float(img[row, col])
    n_pxls_in_compn = 1
    while q:
        last = q.popleft()
        neighb_row = np.arange(max(last[0] - radius, 0),
                               min(last[0] + radius + 1, img.shape[0]))
        neighb_col = np.arange(max(last[1] - radius, 0),
                               min(last[1] + radius + 1, img.shape[1]))
        if adjacency == 8:
            # search over fully-connected neighborhood
            for nrow in neighb_row:
                for ncol in neighb_col:
                    #print(nrow, ncol)
                    if ans[nrow, ncol] == -1 and \
                        abs(img[nrow, ncol] * n_pxls_in_compn - compn_sum) < tol * n_pxls_in_compn:
                            # abs(img[nrow, ncol] - compn_sum / n_pxls_in_compn) < tol
                        #abs(img[nrow, ncol] - img[last]) < tol:
                        ans[nrow, ncol] = cluster
                        q.append((nrow, ncol))
        elif adjacency == 4:
            # search over row and column of specified radius
            for nrow in neighb_row:
                ncol = last[1]
                if ans[nrow, ncol] == -1 and \
                    abs(img[nrow, ncol] * n_pxls_in_compn - compn_sum) < tol * n_pxls_in_compn:
                    #abs(img[nrow, ncol] - img[last]) < tol:
                    ans[nrow, ncol] = cluster
                    q.append((nrow, ncol))
            for ncol in neighb_col:
                nrow = last[0]
                if ans[nrow, ncol] == -1 and \
                    abs(img[nrow, ncol] * n_pxls_in_compn - compn_sum) < tol * n_pxls_in_compn:
                    #abs(img[nrow, ncol] - img[last]) < tol:
                    ans[nrow, ncol] = cluster
                    q.append((nrow, ncol))
        else:
            raise imh.ImLibException('adjacency parameter can take only 4 or 8 value')
        
    return ans


def leave_segments(label, cl_no, replace_value=0):
    ans = label.copy()
    for i in xrange(label.shape[0]):
        for j in xrange(label.shape[1]):
            if label[i, j] not in cl_no:    
                ans[i, j] = replace_value
    return ans


def binarize_hand_img(gb):
    ans = np.full_like(gb, -1, dtype=np.int64)
    ans = expand(gb, ans, 0, 0, 0, radius=1, tol=40, adjacency=4)
    ans[ans == -1] = 1
    
    label = skimage.measure.label(ans)
    max_area_compn = np.argmax(map(attrgetter('area'), skimage.measure.regionprops(label)))
    ans2 = leave_segments(label, [max_area_compn + 1])
    return ans2 != 0


def aligned_angle(vec):
    x, y = vec[1], -vec[0]    # converting (i, j) to (x, y)
    atan2 = np.arctan2(y, x)
    if atan2 < 0:
        atan2 += 2 * np.pi
    # Transforming range of angles, making use of fact that hand is oriented upwards
    if atan2 > 3 * np.pi / 2.0:
        atan2 -= 3 * np.pi / 2.0
    else:
        atan2 += np.pi / 2.0
    return atan2


def get_fingertips_and_valleys(binarized, return_center=False):
    binarized_1 = skimage.morphology.closing(binarized, selem=np.ones((2, 2))).astype(int)
    
    # Finding center
    med, dt = skimage.morphology.medial_axis(binarized, return_distance=True)
    center = (np.argmax(dt.ravel()) / dt.shape[1], np.argmax(dt.ravel()) % dt.shape[1])
    ci, cj = center
    
    # Finding contours
    contours = cv2.findContours(binarized.copy().astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    contours = contours[0]
    for contour in contours:
        for i in xrange(len(contour)):
            contour[i, 0, 0], contour[i, 0, 1] = contour[i, 0, 1], contour[i, 0, 0]
        if cv2.pointPolygonTest(contour, center, False) >= 0:
            contours = contour
            break
    else:
        print 'No contour contains center of hand'
        contours = contours[0]
    
    contours_list = contours.reshape((contours.shape[0], 2))
    
    # Finding fingertips
    hull_list_initial = cv2.convexHull(contours_list)
    hull_list_initial = hull_list_initial.reshape((hull_list_initial.shape[0], 2))
    hull_list = hull_list_initial.copy()

    side_gap = 5    # ignoring points that lie just near image border
    sz = binarized.shape
    hull_list = hull_list[(side_gap < hull_list[:, 0]) & (hull_list[:, 0] < (sz[0] - side_gap)) & \
                          (side_gap < hull_list[:, 1]) & (hull_list[:, 1] < (sz[1] - side_gap))]

    fingertips = []
    fingertip_radius = 60
    fingertip_farthest = None
    for i in xrange(5):
        if hull_list.size > 0:
            argmax = np.argmax((hull_list[:, 0] - ci) ** 2 + (hull_list[:, 1] - cj) ** 2)
            max_pnt = hull_list[argmax]
            hull_list = hull_list[(hull_list[:, 0] - max_pnt[0]) ** 2 + \
                                  (hull_list[:, 1] - max_pnt[1]) ** 2 >= fingertip_radius ** 2]
            if fingertip_farthest is None:
                fingertip_farthest = max_pnt
                dot = np.dot(hull_list - center, max_pnt - center)
                cross = np.cross(hull_list - center, max_pnt - center)
                hull_list = hull_list[(dot >= 0) | \
                                      ((dot < 0) & (np.abs(cross) >= np.linalg.norm(hull_list - center, axis=1) * \
                                       np.linalg.norm(max_pnt - center) * np.sin(np.pi / 2.0 + np.pi / 10.0)))]
            fingertips.append(max_pnt)
    
    # Finding valleys
    hull_idx = [np.where((contours_list[:, 0] == pnt[0]) & (contours_list[:, 1] == pnt[1]))[0][0]
            for pnt in hull_list_initial]
    defects = cv2.convexityDefects(contours_list, np.array(hull_idx))
    defects_idx = defects[:, :, 2].ravel()
    defects_depth = defects[:, :, 3].ravel()
    if fingertip_farthest is not None:
        dot = np.dot(contours_list[defects_idx] - center, fingertip_farthest - center)
        cross = np.cross(contours_list[defects_idx] - center, fingertip_farthest - center)
        defects_depth = defects_depth[(dot >= 0) | \
                                      (np.abs(cross) >= np.linalg.norm(contours_list[defects_idx] - center, axis=1) * \
                                       np.linalg.norm(fingertip_farthest - center) * np.sin(np.pi / 2.0 + np.pi / 8.0))]
        defects_idx = defects_idx[(dot >= 0) | \
                                  (np.abs(cross) >= np.linalg.norm(contours_list[defects_idx] - center, axis=1) * \
                                   np.linalg.norm(fingertip_farthest - center) * np.sin(np.pi / 2.0 + np.pi / 8.0))]
    chosen_defects_idx = defects_idx[np.argsort(defects_depth)[-4:]]
    
    valleys = []
    for idx in chosen_defects_idx:
        pnt = contours_list[idx]
        valleys.append(pnt)
    
    if return_center:
        return fingertips, valleys, center
    return fingertips, valleys


def get_features(binarized, fingertips, valleys, center):
    bin_w_key_pts = np.zeros((binarized.shape[0] * 3, binarized.shape[1] * 3), dtype=np.int64)
    bin_w_key_pts[binarized.shape[0]:2 * binarized.shape[0], binarized.shape[1]:2 * binarized.shape[1]] = \
        binarized.copy()
    for pnt in fingertips:
        bin_w_key_pts[binarized.shape[0] + pnt[0] - 5:binarized.shape[0] + pnt[0] + 5, \
                      binarized.shape[1] + pnt[1] - 5:binarized.shape[1] + pnt[1] + 5] = 5
    for pnt in valleys:
        bin_w_key_pts[binarized.shape[0] + pnt[0] - 5:binarized.shape[0] + pnt[0] + 5, \
                      binarized.shape[1] + pnt[1] - 5:binarized.shape[1] + pnt[1] + 5] = 10
    fingertips_dist_sq = map(lambda pnt: ((pnt - np.array(center)) ** 2).sum(), fingertips)
    fingertip_farthest = fingertips[np.argmax(fingertips_dist_sq)]
    rot_angle = np.degrees(np.pi - aligned_angle(fingertip_farthest - center))
    bin_rot = sp.misc.imrotate(bin_w_key_pts, rot_angle, interp='nearest')
    
    # Finding center after rotation
    med, dt = skimage.morphology.medial_axis(bin_rot, return_distance=True)
    center_rot = (np.argmax(dt.ravel()) / dt.shape[1], np.argmax(dt.ravel()) % dt.shape[1])

    # Acquiring fingertips from rotated image and sorting them by angle
    fingertips_label = skimage.measure.label(bin_rot == 127, neighbors=8, connectivity=3)
    fingertips_rot = map(attrgetter('centroid'), skimage.measure.regionprops(fingertips_label))
    fingertips_rot = np.array(fingertips_rot, dtype=np.int64)
    fingertips_angles = np.array(map(lambda pnt: aligned_angle(pnt - center_rot), fingertips_rot))
    fingertips_rot = fingertips_rot[np.argsort(fingertips_angles)]

    # Acquiring valleys from rotated image and sorting them by angle
    valleys_label = skimage.measure.label(bin_rot == 255, neighbors=8, connectivity=3)
    valleys_rot = map(attrgetter('centroid'), skimage.measure.regionprops(valleys_label))
    valleys_rot = np.array(valleys_rot, dtype=np.int64)
    valleys_angles = np.array(map(lambda pnt: aligned_angle(pnt - center_rot), valleys_rot))
    valleys_rot = valleys_rot[np.argsort(valleys_angles)]

    if len(fingertips_rot) < 5 or len(valleys_rot) < 4:
        return None
    key_points = []
    for i in xrange(4):
        key_points.extend([fingertips_rot[i], valleys_rot[i]])
    key_points.append(fingertips_rot[4])
    features = np.diff(key_points, axis=0)
    features = np.sqrt(features[:, 0] ** 2 + features[:, 1] ** 2)
    return features