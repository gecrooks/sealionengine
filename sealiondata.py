#!/usr/bin/env python

"""Sea Lion Prognostication Engine

https://www.kaggle.com/c/noaa-fisheries-steller-sea-lion-population-count

https://github.com/gecrooks/sealionengine
"""
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import sys
import os
from collections import namedtuple
from collections import OrderedDict
import argparse
import random
import csv

import numpy as np

import PIL
from PIL import Image

import skimage
import skimage.io
import skimage.draw
import skimage.measure

import shapely
import shapely.geometry
from shapely.geometry import Polygon

# To actiate multiprocessor support, install pathos.
try:
    from pathos.multiprocessing import ProcessingPool as Pool
    MULTIPROCESSING = True
except ImportError:
    MULTIPROCESSING = False

random.seed(42)

# Notes
# cls -- sea lion class
# tid -- train, train dotted, or test image id
# _nb -- abbreviation for number
#
# row, col, ch -- Image arrays are indexed as (rows, columns, channels) with origin at top left.
#                   Beware: Some libraries use (x,y) cartesian coordinates (e.g. cv2, matplotlib)
#                   Channels are in RGB order. Beware: openCV uses BGR order (!?)
# rr, cc -- lists of row and column coordinates
#
# By default, SeaLionData expects source data to be located in ../input, and saves processed data
# to ./outdir
#
# With contributions from Kagglers @bitsofbits, @authman, @mfab, @depthfirstsearch, @JandJ, 
# @LivingProgram, @mrgloom ...
#




# ================ Meta ====================
__description__ = 'Sea Lion Prognostication Engine'
__version__ = '0.4.0'
__license__ = 'MIT'
__author__ = 'Gavin Crooks (@threeplusone)'
__status__ = "Prototype"
__copyright__ = "Copyright 2017"

# python -c 'import sealiondata; sealiondata.package_versions()'
def package_versions():
    print('sealionengine \t', __version__)
    print('python        \t', sys.version[0:5])
    print('numpy         \t', np.__version__)
    print('skimage       \t', skimage.__version__)
    print('pillow (PIL)  \t', PIL.__version__)
    print('shapely       \t', shapely.__version__)

    if MULTIPROCESSING:
        import pathos
        print('pathos       \t', pathos.__version__)
        import dill
        print('dill         \t', dill.__version__)

SOURCEDIR = os.path.join('..', 'input')

OUTDIR = os.path.join('.', 'outdir')

TILE_SIZE = 128   # Default tile size

VERBOSITY = namedtuple('VERBOSITY', ['QUITE', 'NORMAL', 'VERBOSE', 'DEBUG'])(0, 1, 2, 3)

SeaLionCoord = namedtuple('SeaLionCoord', ['tid', 'cls', 'row', 'col'])

TileCoord = namedtuple('TileCoord', ['tid', 'row', 'row_stop', 'col', 'col_stop'])


class SeaLionData(object):

    def __init__(self, sourcedir=SOURCEDIR, outdir=OUTDIR, verbosity=VERBOSITY.NORMAL):
        self.sourcedir = sourcedir
        self.outdir = outdir
        self.verbosity = verbosity

        self.cls_nb = 5

        self.cls_names = (
            'adult_males',
            'subadult_males',
            'adult_females',
            'juveniles',
            'pups',
            'NOT_A_SEA_LION')

        self.cls_idx = { cls_name: n for n, cls_name in enumerate(self.cls_names)}


        # backported from @bitsofbits. Average actual color of dot centers.
        self.cls_colors = (
            (243, 8, 5),          # red
            (244, 8, 242),        # magenta
            (87, 46, 10),         # brown (Brown sea lions on brown rocks marked with brown dots!)
            (25, 56, 176),        # blue
            (38, 174, 21),        # green
            )


        self.dot_radius = 3

        self.train_nb = 948

        self.test_nb = 18636

        self.paths = {
            # Source paths
            'sample'     : os.path.join(sourcedir, 'sample_submission.csv'),
            'counts'     : os.path.join(sourcedir, 'Train', 'train.csv'),
            'train'      : os.path.join(sourcedir, 'Train', '{tid}.jpg'),
            'dotted'     : os.path.join(sourcedir, 'TrainDotted', '{tid}.jpg'),
            'test'       : os.path.join(sourcedir, 'Test', '{tid}.jpg'),
            # Data paths
            'coords'     : os.path.join(outdir, 'coords.csv'),
            'chunk'      : os.path.join(outdir, ' {tid}_{cls}_{row}_{col}_{size}.png'),
            }

        self.bad_train_ids = (
            # From MismatchedTrainImages.txt
            3,       # Region mismatch
            # 7,     # TrainDotted rotated 180 degrees. Hot patch in load_dotted_image()
            9,       # Region mismatch
            21,      # Region mismatch
            30,      # Exposure mismatch -- not fixable
            34,      # Exposure mismatch -- not fixable
            71,      # Region mismatch
            81,      # Region mismatch
            89,      # Region mismatch
            97,      # Region mismatch
            151,     # Region mismatch
            184,     # Exposure mismatch -- almost fixable
            # 215,   # TrainDotted rotated 180 degrees. Hot patch in load_dotted_image()
            234,     # Region mismatch
            242,     # Region mismatch
            268,     # Region mismatch
            290,     # Region mismatch
            311,     # Region mismatch
            # 331,   # TrainDotted rotated 180 degrees. Hot patch in load_dotted_image()
            # 344,   # TrainDotted rotated 180 degrees. Hot patch in load_dotted_image()
            380,     # Exposure mismatch -- not fixable
            384,     # Region mismatch
            # 406,   # Exposure mismatch -- fixed by find_coords()
            # 421,   # TrainDotted rotated 180 degrees. Hot patch in load_dotted_image()
            # 469,   # Exposure mismatch -- fixed by find_coords()
            # 475,   # Exposure mismatch -- fixed by find_coords()
            490,     # Region mismatch
            499,     # Region mismatch
            507,     # Region mismatch
            # 530,   # TrainDotted rotated. Hot patch in load_dotted_image()
            531,     # Exposure mismatch -- not fixable
            # 605,   # In MismatchedTrainImages, but appears to be O.K.
            # 607,   # Missing annotations on 2 adult males, added to missing_coords
            614,     # Exposure mismatch -- not fixable
            621,     # Exposure mismatch -- not fixable
            # 638,   # TrainDotted rotated. Hot patch in load_dotted_image()
            # 644,   # Exposure mismatch, but not enough to cause problems
            687,     # Region mismatch
            712,     # Exposure mismatch -- not fixable
            721,     # Region mismatch
            767,     # Region mismatch
            779,     # Region mismatch
            # 781,   # Exposure mismatch -- fixed by find_coords()
            # 794,   # Exposure mismatch -- fixed by find_coords()
            800,     # Region mismatch
            811,     # Region mismatch
            839,     # Region mismatch
            840,     # Exposure mismatch -- not fixable
            869,     # Region mismatch
            # 882,   # Exposure mismatch -- fixed by find_coords()
            # 901,   # Train image has (different) mask already, but not actually a problem
            903,     # Region mismatch
            905,     # Region mismatch
            909,     # Region mismatch
            913,     # Exposure mismatch -- not fixable
            927,     # Region mismatch
            946,     # Exposure mismatch -- not fixable

            # Additional anomalies
            857,     # Missing annotations on all sea lions (Kudos: @depthfirstsearch)
        )

        # A few TrainDotted images are rotated relative to Train.
        # Hot patch in load_dotted_image()
        # Number of 90 degree rotations to apply.
        self.dotted_rotate = {7:2, 215:2, 331:2, 344:2, 421:2, 530:1, 638:1}

        bad_coords = (
            (83, 2, 46, 4423),      # Empty sea?
            (100, 3, 1170, 3111),   # Rock
            (104, 3, 1041, 4093),   # Rock
            # (105, 3, 1154, 1954),   # Not centered.
            
            # 200: lots of pups marked as adults.
            (200, 0, 1129, 4257), (200, 0, 1266, 3877), (200, 0, 1290, 4117), (200, 0, 1337, 4443),
            (200, 0, 1364, 4884), (200, 0, 1452, 4621), (200, 0, 1498, 4208), (200, 0, 1534, 3837),
            (200, 0, 1565, 4965), (200, 0, 1570, 4263), (200, 0, 1684, 4824), (200, 0, 1854, 4528),
            (200, 0, 1999, 4958), (200, 0, 2001, 4849), (200, 0, 2008, 4566), (200, 0, 2051, 4434),
            (200, 0, 2070, 4552), (200, 0, 2170, 4637), (200, 0, 2222, 4794), (200, 0, 2272, 4355),
            (200, 0, 637, 3901), (200, 0, 718, 3544), (200, 0, 843, 3986), (200, 0, 901, 4301),
            (200, 2, 1547, 4982),           
            
            (259, 0, 1358, 2228),   # Empty sea (kudos: @authman)
            (275, 0, 272, 4701),    # Empty sea (kudos: @authman)
            (292, 2, 4, 248),       # Rock
            (303, 3, 1533, 3337),   # Rock
            (323, 0, 2223, 2022),   # Empty sea (Kudos: @A3DD)
            (323, 0, 2498, 2352),   # Empty sea (Kudos: @A3DD)
            (658, 0, 900, 4149),    # (Kudos: @A3DD)
            (658, 0, 1657, 4473),   # (Kudos: @A3DD)
            (658, 0, 2765, 4572),   # (Kudos: @A3DD)
            (691, 0, 2576, 3294),   # Empty sea (Kudos: @A3DD)
            (741, 0, 1418, 3258),   # Empty sea (kudos: @authman)
            (741, 0, 2466, 3700),   # Empty sea (kudos: @authman)
            (912, 2, 813, 3117),    # Random dot on tail of adult male
            (921, 3, 2307, 1418),   # Empty sea
            (921, 3, 2351, 1398),   # Empty sea
            (933, 3, 3177, 3297),   # Sea
            (934, 2, 1148, 4094),   # Off center    (TODO: CHECK)
            (939, 2, 2569, 2957),   # Shadow
            (939, 4, 2756, 2176),   # Shadow
            (943, 3, 1858, 2711),   # Rock
            
            # Kudos: @LivingProgram
            # https://github.com/LivingProgram/kaggle-sea-lion-data/changes.csv
            (292, 3, 336, 7), (323, 0, 2223, 2022), (323, 0, 2499, 2353), (523, 0, 2148, 5615),
            (587, 4, 122, 1708), (587, 4, 150, 1667), (587, 4, 195, 1665), (587, 4, 210, 1684),
            (587, 4, 174, 1554), (658, 0, 900, 4149), (658, 0, 1657, 4473), (658, 0, 2764, 4572),
            (827, 0, 2980, 1589), (827, 0, 2496, 2024), (827, 0, 2646, 2669), (827, 0, 1679, 2873),
            (941, 0, 1109, 4405), (941, 0, 1768, 4126), (941, 0, 1991, 4948), (941, 0, 2005, 4765),
            (941, 0, 2291, 3710),
            
        )
        self.bad_coords = to_tid_coords(bad_coords)

        missing_coords = (
            (15, 2, 1686, 2620),    # Merged double dot, both rejected
            (148, 1, 1390, 4525),
            
            # 200: lots of pups marked as adults.
            (200, 4, 1129, 4257), (200, 4, 1266, 3877), (200, 4, 1290, 4117), (200, 4, 1337, 4443),
            (200, 4, 1364, 4884), (200, 4, 1452, 4621), (200, 4, 1498, 4208), (200, 4, 1534, 3837),
            (200, 4, 1565, 4965), (200, 4, 1570, 4263), (200, 4, 1684, 4824), (200, 4, 1854, 4528),
            (200, 4, 1999, 4958), (200, 4, 2001, 4849), (200, 4, 2008, 4566), (200, 4, 2051, 4434),
            (200, 4, 2070, 4552), (200, 4, 2170, 4637), (200, 4, 2222, 4794), (200, 4, 2272, 4355),
            (200, 4, 637, 3901), (200, 4, 718, 3544), (200, 4, 843, 3986), (200, 4, 901, 4301),
            
            (607, 0, 1160, 2459),
            (607, 0, 1245, 2836),
            (691, 0, 114, 1539),    # (Kudos: @A3DD)
            (816, 3, 2256, 767),
            (899, 2, 550, 2114),    # adult_female or juvenile?
            (934, 2, 1134, 4096),   # Off center
            (934, 2, 1086, 4528),

            # Kudos: @LivingProgram
            # https://github.com/LivingProgram/kaggle-sea-lion-data/changes.csv
            # (Many additions from changes.csv are due to errors in dot extraction, not needed)
            (58, 0, 2536, 2287), (58, 0, 2549, 1950), (66, 0, 3040, 277), (80, 2, 414, 1697),
            (170, 0, 623, 3234), (177, 1, 2630, 5186), (187, 2, 1931, 3017), (197, 2, 2028, 2676),
            (235, 0, 193, 2312), (235, 0, 369, 2240), (235, 0, 285, 2388), (235, 0, 177, 2758),
            (235, 0, 457, 3094), (235, 0, 719, 2866), (235, 0, 831, 2600), (235, 0, 309, 2111),
            (235, 0, 693, 2141), (235, 0, 179, 3362), (235, 0, 351, 3540), (235, 0, 543, 3504),
            (235, 0, 213, 3840), (235, 0, 861, 3706), (235, 0, 849, 2731), (235, 0, 1061, 2559),
            (235, 0, 891, 2156), (235, 0, 1189, 2000), (235, 0, 2241, 1330),(235, 0, 2039, 2502),
            (235, 0, 1847, 3234), (235, 0, 1149, 3560), (235, 0, 937, 4082), (235, 0, 1555, 3939),
            (235, 0, 2009, 2354), (235, 0, 641, 5470), (235, 0, 888, 5594), (235, 0, 1325, 4327),
            (235, 0, 1869, 4533), (240, 2, 3336, 3572), (310, 3, 2888, 4715), (323, 0, 1681, 617),
            (323, 0, 1960, 955), (323, 0, 2048, 4853), (328, 0, 700, 1175), (328, 0, 500, 1392),
            (338, 0, 3579, 1846), (359, 1, 2870, 4813), (361, 0, 3230, 222), (368, 2, 1057, 2221),
            (368, 0, 513, 2023), (395, 2, 1237, 2579), (395, 2, 1181, 2873), (403, 2, 1505, 3668),
            (431, 3, 2368, 3223), (431, 1, 2566, 2375), (437, 2, 759, 3589), (437, 2, 2439, 4943),
            (437, 2, 2734, 4727), (462, 2, 2103, 4014), (538, 2, 1348, 1552), (538, 2, 956, 1583),
            (578, 4, 2197, 2131), (578, 2, 2187, 2145), (658, 0, 3047, 4597), (658, 0, 2441, 3980),
            (658, 0, 1804, 4502), (658, 0, 1319, 3894), (658, 0, 581, 3563), (684, 2, 1676, 2866),
            (741, 0, 421, 1802), (759, 2, 1332, 2127), (803, 2, 3349, 1356), (827, 0, 1677, 1931),
            (827, 0, 2680, 1734), (827, 0, 3002, 665), (827, 0, 2493, 1073), (881, 2, 2994, 2538),
            (881, 2, 2818, 2412), (881, 0, 2315, 2203), (881, 2, 1946, 2166), (881, 2, 1766, 1686),
            (912, 2, 2287, 2760), (912, 2, 1601, 2898), (912, 2, 1280, 2616), (912, 2, 1261, 2547),
            (924, 2, 1600, 736), (939, 2, 988, 2317), (939, 2, 519, 2521), (941, 0, 1159, 1378),
            (941, 0, 1507, 2420), (941, 0, 1802, 1101), (941, 0, 2049, 1705), (941, 0, 2029, 1903),
            (941, 0, 2351, 679),
        )
        self.missing_coords = to_tid_coords(missing_coords)


        # Corrections to train.csv counts
        # (For about 20 of these, kudos: @authman)
        self.better_counts = {
            2 : (2, 0, 37, 19, 0),  11 : (3, 5, 36, 13, 0), 13 : (1, 5, 20, 13, 0),
            15 : (2, 3, 33, 56, 0), 18 : (2, 3, 0, 0, 0), 36 : (8, 17, 0, 0, 0),
            38 : (3, 0, 33, 0, 0), 40 : (2, 2, 62, 7, 0), 47 : (13, 14, 48, 3, 33),
            52 : (2, 3, 20, 23, 0), 66 : (8, 5, 23, 17, 2), 83 : (5, 2, 44, 41, 0),
            148: (0, 3, 0, 5, 0), 200: (12, 3, 55, 2, 29), 221: (6, 1, 26, 9, 2),
            292: (5, 5, 49, 42, 1), 299: (27, 9, 209, 32, 55), 312: (1, 1, 21, 14, 0),
            323: (2, 0, 1, 0, 1), 335: (6, 36, 18, 12, 0), 426: (2, 6, 11, 42, 5),
            479: (5, 4, 0, 0, 0), 492: (2, 1, 9, 21, 1), 510: (5, 1, 0, 0, 0),
            529: (5, 2, 15, 12, 0), 538: (10, 2, 162, 9, 115), 577: (3, 1, 116, 97, 0),
            593: (1, 2, 32, 58, 0), 607: (2, 3, 14, 3, 0), 643: (1, 5, 0, 21, 0),
            698: (1, 0, 0, 14, 0), 706: (2, 4, 39, 18, 0), 707: (4, 21, 1, 7, 0),
            776: (8, 2, 25, 2, 29), 845: (8, 1, 9, 17, 0), 899: (2, 2, 4, 3, 0),
            912: (30, 2, 247, 13, 205), 933: (6, 7, 68, 68, 21), 943: (1, 1, 24, 25, 0),
        }

        # Extra areas to mask out. tid: list_of_circles (row, col, radius)
        self.extra_masks = { 
            129: ((2915, 3057, 500), ), # Large group bottom middle are not marked
            759: ((1571, 3275, 100), ), # DEAD written in red ink on a sea lion in the center.
        }


        # Further notes on images of note. Training images that have weird stuff in them.
        # 50    Unlabeled things are northern fur seals, not sea lions (Kudos: Katie @sweenkl)
        # 122   A juvenile bottom right: Most of lion and blue dot have
        #         been obliterated by black mask. Should not include in train set?
        # 280   Similar issue to 695. In fact, could be same beach. (kudos: @pmgpmg)
        # 338   Red lines on dotted. Appears to be crossing out some 'lions. Should
        #         have been a masked area? One obvious unmarked adult male just above red area.
        # 513   Seals?
        # 695 	Lots of unmarked animals. Sea lions seem to favor rocks, not sand.
        #        Probably northern fur seals. Possible Bogoslof island. (Kudos: Katie @sweenkl)
        #        See also 280
        # 803   Big bloody red mark at bottom middle!?
        # 895   Same rocks as 907 (but taken at different times) (Kudos: toshi_k)
        # 907   See 895
        # 912   Large red box and arrow on the train image.
        # 934   Seals (?) on rock north of sea lions.
        # 935   Only one sea lion in entire image
        # 945   Beach. Sea Lions seem to normally favor rocks.

        # caches
        self._tid_counts = None
        self._tid_coords = None



    @property
    def trainshort1_ids(self):
        tids = list(range(0, 11))
        tids = self._remove_bad_ids(tids)
        return tids


    @property
    def trainshort2_ids(self):
        tids = list(range(41, 51))
        tids = self._remove_bad_ids(tids)
        return tids


    @property
    def train_ids(self):
        """List of all valid train ids"""
        tids = list(range(0, self.train_nb))
        tids = self._remove_bad_ids(tids)
        return tids


    def _remove_bad_ids(self, tids):
        tids = list(set(tids) - set(self.bad_train_ids))
        tids.sort()
        return tids


    @property
    def test_ids(self):
        return list(range(0, self.test_nb))


    def path(self, name, **kwargs):
        """Return path to various source files"""
        path = self.paths[name].format(**kwargs)
        return path


    @property
    def tid_counts(self):
        """A map from train_id to list of sea lion class counts"""
        if self._tid_counts is None:
            tid_counts = OrderedDict()
            fn = self.path('counts')
            with open(fn) as f:
                f.readline()
                for line in f:
                    counts = tuple(map(int, line.split(',')))
                    tid_counts[counts[0]] = counts[1:]
            # Apply corrections
            for tid, counts in self.better_counts.items():
                tid_counts[tid] = counts
            self._tid_counts = tid_counts
        return self._tid_counts


    def count_coords(self, tid_coords):
        """Return a map from ids to list of class counts.

        Args:
            tid_coords: A map from ids to coordinates.

        Returns:
            A list of integer sea lion class counts

        """
        tid_counts = OrderedDict()
        for tid, coords in tid_coords.items():
            counts = [0]*self.cls_nb
            for tid, cls, row, col in coords:
                counts[cls] += 1
            tid_counts[tid] = counts
        return tid_counts


    def rmse(self, tid_counts):
        error = np.zeros(shape=[self.cls_nb])
        err_nb = 0

        self._progress('\ntid \t true_count     \t obs_count       \t difference',
                       end='\n', verbosity=VERBOSITY.VERBOSE)

        for tid in tid_counts:
            true_counts = self.tid_counts[tid]
            obs_counts = tid_counts[tid]
            diff = np.asarray(true_counts) - np.asarray(obs_counts)
            err_nb += np.count_nonzero(diff)
            error += diff*diff

            if diff.any():
                self._progress('{} \t{} \t{} \t{}'.format(tid, true_counts, obs_counts, diff),
                               end='\n', verbosity=VERBOSITY.VERBOSE)

        error /= len(tid_counts)
        rmse = np.sqrt(error).sum() / self.cls_nb
        error_fraction = err_nb / (len(tid_counts)* self.cls_nb)

        return rmse, error_fraction


    def load_train_image(self, train_id, scale=1, border=0, mask=False):
        """Return image as numpy array.

        Args:
            border (int): Add a black border of this width around image
            mask (bool): If true copy masks from corresponding dotted image

        Returns:
            uint8 numpy array
        """
        img = self._load_image('train', train_id, scale, border)
        if mask:
            # The masked areas are not uniformly black, presumable due to
            # jpeg compression artifacts
            MASK_MAX = 40
            dot_img = self.load_dotted_image(train_id, scale, border).astype(np.uint16).sum(axis=-1)
            img = np.copy(img)
            img[dot_img < MASK_MAX] = 0
        return img


    def load_dotted_image(self, train_id, scale=1, border=0, circled=False):
        img = self._load_image('dotted', train_id, scale, border)

        if train_id in self.extra_masks:
            for row, col, radius in self.extra_masks[train_id]:
                rr, cc = skimage.draw.circle(row, col, radius, shape = img.shape)
                img = np.copy(img)
                img[rr, cc] = (0, 0, 0)

        # When dotted image is rotated relative to train, apply hot patch. (kudos: @authman)
        if train_id in self.dotted_rotate:
            rot = self.dotted_rotate[train_id]
            img = np.rot90(img, rot)

        if circled: 
            assert scale == 1
            assert border == 0
            img = np.copy(img)
            img = self.draw_circles(np.copy(img), self.tid_coords[train_id])        

        return img


    def load_test_image(self, test_id, scale=1, border=0):
        return self._load_image('test', test_id, scale, border)


    def _load_image(self, itype, tid, scale=1, border=0):
        fn = self.path(itype, tid=tid)

        # Workaround for weird issue in pillow that throws ResourceWarnings
        with open(fn, 'rb') as img_file:
            with Image.open(img_file) as image:
                if scale != 1:
                    width, height = image.size # width x height for PIL
                    image = image.resize((width//scale, height//scale), Image.ANTIALIAS)

                img = np.asarray(image)

        if border:
            height, width, channels = img.shape
            bimg = np.zeros(shape=(height+border*2, width+border*2, channels), dtype=np.uint8)
            bimg[border:-border, border:-border, :] = img
            img = bimg
        return img


    def find_coords(self, train_id):
        """Extract coordinates of dotted sealions from TrainDotted image

        Args:
            train_id:

        Returns:
             list of SeaLionCoord objects
        """

        # Empirical constants
        MIN_DIFFERENCE = 16
        MIN_AREA = 7
        MAX_AREA = 50   # Reduced to 50 from 100 to catch a few weird stray red lines, e.g. 523, 526
        MAX_AVG_DIFF = 50
        MAX_COLOR_DIFF = 32
        MAX_MASK = 8

        # In a few instances, increasing MAX_COLOR_DIFF helps
        # (But if we set this as default we pick up extra spurious dots elsewhere)
        if train_id in [491, 816]:
            MAX_COLOR_DIFF = 48

        src_img = np.asarray(self.load_train_image(train_id, mask=True), dtype=np.float)
        dot_img = np.asarray(self.load_dotted_image(train_id), dtype=np.float)

        # Sometimes the exposure of the Train and TrainDotted images is different.
        # If mismatch is not too bad we can sometimes fix this problem.
        # (see also comments on bad_train_ids)
        ratio = src_img.sum() / dot_img.sum()
        MISMATCHED_EXPOSURE = 1.05
        if ratio > MISMATCHED_EXPOSURE or ratio < 1/MISMATCHED_EXPOSURE:
            self._progress(' (Adjusting exposure: {} {})'.format(train_id, ratio), 
                           verbosity=VERBOSITY.VERBOSE)
            # We adjust the source image so not to mess up dot colors
            src_img /= ratio

        img_diff = np.abs(src_img - dot_img)

        # Detect bad data. If train and dotted images are very different then somethings wrong.
        avg_diff = img_diff.sum() / (img_diff.shape[0] * img_diff.shape[1])
        if avg_diff > MAX_AVG_DIFF:
            self._progress('( Bad train image -- exceeds MAX_AVG_DIFF: {} )'.format(train_id))
            return ()

        img_diff = np.max(img_diff, axis=-1)

        img_diff[img_diff < MIN_DIFFERENCE] = 0
        img_diff[img_diff >= MIN_DIFFERENCE] = 255

        sealions = []

        for cls, color in enumerate(self.cls_colors):
            # color search backported from @bitsofbits.
            # Note that there are large red boxes and arrows in some training images (e.g. 912)
            # The red of these lines (250,0,10) is sufficiently different from dot red that the
            # lines mostly get filtered out.
            color_array = np.array(color)[None, None, :]
            color_diff = dot_img * (img_diff > 0)[:, :, None] - color_array
            has_color = np.sqrt(np.sum(np.square(color_diff), axis=-1)) < MAX_COLOR_DIFF
            contours = skimage.measure.find_contours(has_color.astype(float), 0.5)

            if self.verbosity == VERBOSITY.DEBUG:
                print()
                fn = 'diff_{}_{}.png'.format(train_id, cls)
                print('Saving train/dotted difference: {}'.format(fn))
                Image.fromarray((has_color*255).astype(np.uint8)).save(fn)

            for cnt in contours:
                p = Polygon(shell=cnt)
                area = p.area
                if area > MIN_AREA and area < MAX_AREA:
                    row, col = p.centroid.coords[0] #DANGER: skimage and cv2 coordinates transposed
                    row = int(round(row))
                    col = int(round(col))

                    NO_RED_DOT_CORRECTION = [6,]
                    if cls == self.cls_idx['adult_males'] and train_id not in NO_RED_DOT_CORRECTION:
                        # Sometimes there are ends of red lines poking out from under the black masks
                        # that get mistaken for adult male red dots.
                        dot_region = src_img[row-4: row+5, col-4:col+5]
                        zero_count = dot_region.size - np.count_nonzero(dot_region)

                        if zero_count > MAX_MASK:
                            self._progress(' (Rejecting {} 0 {} {} : {})'
                                .format(train_id, row, col, zero_count), verbosity=VERBOSITY.DEBUG)
                            continue

                    # Remove known bad coordinates
                    if train_id in self.bad_coords:
                        bad_coords = self.bad_coords[train_id]
                        if any([c.cls == cls and abs(c.row-row) < 3 and abs(c.col-col) < 3 for c in bad_coords]):
                            self._progress(' (Removing bad coord: {} {} {} {})'
                                .format(train_id,cls, row, col), verbosity=VERBOSITY.DEBUG)
                            continue
                    #print(train_id, cls, row, col, dot_img[row, col])

                    sealions.append(SeaLionCoord(train_id, cls, row, col))

        if train_id in self.missing_coords:
            sealions.extend(self.missing_coords[train_id])

        if self.verbosity >= VERBOSITY.VERBOSE:
            counts = [0, 0, 0, 0, 0]
            for c in sealions:
                counts[c.cls] += 1
            print()
            print('train_id', 'true_counts', 'counted_dots', 'difference', sep='\t')
            true_counts = self.tid_counts[train_id]
            print(train_id, true_counts, counts, np.array(true_counts) - np.array(counts), sep='\t')

        if self.verbosity == VERBOSITY.DEBUG:
            img = np.copy(self.load_dotted_image(train_id))
            img = self.draw_circles(img, sealions)
            fn = os.path.join(self.outdir, 'cross_{}.png'.format(train_id))
            print('Saving crossed dots: {}'.format(fn))
            Image.fromarray(img).save(fn)

        return sealions

    def draw_circles(self, img, coords) :
        radius = self.dot_radius*3
        for tid, cls, row, col in coords:
            rr, cc = skimage.draw.circle_perimeter(row, col, radius, shape = img.shape)
            img[rr, cc] = self.cls_colors[cls]
        return img
        

    def save_coords(self, train_ids=None):
        if train_ids is None:
            train_ids = self.train_ids
        fn = self.path('coords')
        self._progress('Saving sea lion coordinates to {}'.format(fn))
        if os.path.exists(fn):
            raise IOError('Output file exists: {}'.format(fn))

        # Multiprocessing support (Kudos: @JandJ)
        if MULTIPROCESSING:
            all_coord_list = Pool().map(self.find_coords, train_ids)
        else:
            all_coord_list = map(self.find_coords, train_ids)

        with open(fn, 'w') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(SeaLionCoord._fields)
            for coord in all_coord_list:
                for c in coord:
                    writer.writerow(c)
        self._progress('done')


    @property
    def tid_coords(self):
        """Loads the coordinates saved by save_coords()
        Returns:
            a dictionary from tid to SeaLionCoords
        """
        if self._tid_coords is None:
            fn = self.path('coords')
            if not os.path.exists(fn): 
                self.save_coords()

            self._progress('( Loading sea lion coordinates from {}'.format(fn))
            with open(fn) as f:
                f.readline()
                slc = [SeaLionCoord(*[int(n) for n in line.split(',')]) for line in f]
            self._progress(')')
            self._tid_coords = to_tid_coords(slc)

        return self._tid_coords

    # python -c 'import sealiondata as sld; sld.SeaLionData().save_sea_lions(100, dotted=True)'
    def save_sea_lions(self, train_id, coords=None, size=TILE_SIZE, dotted=False):
        """Save image chunks of given size centered on sea lion coordinates.
        If no coordinates given, then load training set coordinates.

        Args:
            train_id:
            coords: list of SeaLionCoords
            size: (int) The height and width of each chunk
            dotted: (bool) if true extract chunks from TrainDotted
        """
        self._progress('Saving image chunks...')
        self._progress('\n', verbosity=VERBOSITY.VERBOSE)

        if coords is None:
            coords = self.tid_coords[train_id]

        if dotted:
            img = self.load_dotted_image(train_id, border=size//2)
        else:
            img = self.load_train_image(train_id, border=size//2, mask=True)

        for tid, cls, row, col in coords:
            assert tid == train_id
            fn = self.path('chunk', size=size, tid=tid, cls=cls, row=row, col=col)
            self._progress(' Saving '+fn, end='\n', verbosity=VERBOSITY.VERBOSE)
            Image.fromarray(img[row:row+size, col:col+size, :]).save(fn)
            self._progress()

        self._progress('done')


    def _progress(self, string=None, end=' ', verbosity=VERBOSITY.NORMAL):
        if self.verbosity < verbosity:
            return
        if not string:
            print('.', end='')
        elif string == 'done':
            print(' done')
        else:
            print(string, end=end)
        sys.stdout.flush()

# end SeaLionData

# Utility routines
def dump_namedtuple(filename, tuple_type, list_of_namedtuples):
    with open(filename, 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(tuple_type._fields)
        for item in list_of_namedtuples:
            writer.writerow(item)


# Round up to next size
def roundup(x, size):
    return ((x+size-1) // size) * size


# Round down to previous size
def rounddown(x, size):
    return roundup(x-size+1, size)


def to_tid_coords(coords):
    """Convert list of SeaLionCoords to map from tid to lists of coords"""
    tid_coords = OrderedDict()
    for c in coords:
        if not isinstance(c, SeaLionCoord):
            c = SeaLionCoord(*c)
        tid = c.tid
        if tid not in tid_coords:
            tid_coords[tid] = []
        tid_coords[tid].append(c)
    return tid_coords



# ---------- Command Line Interface ----------
def _cli():

    parser = argparse.ArgumentParser(
        description=__description__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    cmdparser = parser.add_subparsers(
        title='Commands',
        description=None,
        help="-h, --help Additional help",)


    parser.add_argument('--version', action='version', version=__version__)

    parser.add_argument('-s', '--sourcedir', action='store', dest='sourcedir', 
                        default=SOURCEDIR, metavar='PATH', help='Location of input data')

    parser.add_argument('-o', '--outdir', action='store', dest='outdir',
                        default=OUTDIR, metavar='PATH', help='Location of processed data')

    parser.add_argument('-q', '--quite', action='store_const', dest='verbosity',
                        default=VERBOSITY.NORMAL, const=VERBOSITY.QUITE)

    parser.add_argument('-v', '--verbose', action='store_const', dest='verbosity',
                        default=VERBOSITY.NORMAL, const=VERBOSITY.VERBOSE)

    parser.add_argument('-d', '--debug', action='store_const', dest='verbosity',
                        default=VERBOSITY.NORMAL, const=VERBOSITY.DEBUG)


    # == Command: save sea lion coordinates ==
    def coords(sld, **args):
        sld.save_coords()

        if sld.verbosity == VERBOSITY.QUITE:
            return

        # Error analysis
        tid_counts = sld.count_coords(sld.tid_coords)
        rmse, frac = sld.rmse(tid_counts)

        print()
        print('RMSE: {}'.format(rmse))
        print('Error frac: {}'.format(frac))

    p = cmdparser.add_parser('coords', help='Extract and save sea lion coordinates')
    p.set_defaults(func=coords)


    # == Command: save sea lion chunks ==
    def chunks(sld, tids, dotted):
        for tid in tids:
            sld.save_sea_lions(tid, dotted=dotted)

    p = cmdparser.add_parser('chunks', help='Extract and save sea lion chunks')
    p.set_defaults(func=chunks)
    p.add_argument('--dotted', action='store_true', default=False)
    p.add_argument('tids', action='store', nargs='+', type=int)


    # == Command: save circled sea lions ==
    def circled(sld, tids):
        for tid in tids:
            img = sld.load_dotted_image(tid, circled=True)
            fn = os.path.join(sld.outdir, 'circled_{}.png'.format(tid))
            print('Saving circled sealions: {}'.format(fn))
            Image.fromarray(img).save(fn)

    p = cmdparser.add_parser('circled', help='Generate image with known sea lions circled.')
    p.set_defaults(func=circled)
    p.add_argument('tids', action='store', nargs='+', type=int)


    # == Command: print installed package versions ==
    def packages(sld):
        package_versions()

    p = cmdparser.add_parser('packages', help='Print installed package versions and exit')
    p.set_defaults(func=packages)


    # Run command
    opts = vars(parser.parse_args())

    sourcedir = opts.pop('sourcedir')
    outdir = opts.pop('outdir')
    verbosity = opts.pop('verbosity')

    sld = SeaLionData(sourcedir=sourcedir, outdir=outdir, verbosity=verbosity)

    func = opts.pop('func')
    func(sld, **opts)


if __name__ == "__main__":
    _cli()
