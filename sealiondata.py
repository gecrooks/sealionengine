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
import random ; random.seed(42)
import operator
import glob
import csv 
from math import sqrt

import numpy as np

import PIL
from PIL import Image, ImageDraw, ImageFilter

import skimage
import skimage.io
import skimage.draw
import skimage.measure

import shapely
import shapely.geometry
from shapely.geometry import Polygon

try :
    from pathos.multiprocessing import ProcessingPool as Pool
    MULTIPROCESSING = True
except ImportError:
    MULTIPROCESSING = False

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
# By default, SeaLionData expects source data to be located in ../input, and saves processed data to ./outdir
#
#
# With contributions from Kaggles @bitsofbits, @authman, @mfab, @depthfirstsearch, @JandJ ...
#




# ================ Meta ====================
__description__ = 'Sea Lion Prognostication Engine'
__version__ = '0.3.0'
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
            
        self.cls_idx = namedtuple('ClassIndex', self.cls_names)(*range(0,6))
        
        # backported from @bitsofbits. Average actual color of dot centers.
        self.cls_colors = (
            (243,8,5),          # red
            (244,8,242),        # magenta
            (87,46,10),         # brown (Brown sea lions on brown rocks marked with brown dots!)
            (25,56,176),        # blue
            (38,174,21),        # green
            )
    
            
        self.dot_radius = 3
        
        self.train_nb = 947
        
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
            'chunk'      : os.path.join(outdir, 'chunk_{tid}_{cls}_{row}_{col}_{size}.png'),
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
            129,     # Raft of marked juveniles in water (middle top). But another 
                     # large group bottom middle are not marked
            200,     # lots of pups marked as adult males
            235,     # None of the 35 adult males have been labelled
            857,     # Missing annotations on all sea lions (Kudos: @depthfirstsearch)
            941,     # 5 adult males not marked    
        )
            
        # A few TrainDotted images are rotated relative to Train.
        # Hot patch in load_dotted_image()
        # Number of 90 degree rotations to apply. 
        self.dotted_rotate = {7:2, 215:2, 331:2, 344:2, 421:2, 530:1, 638:1}
           
        bad_coords = (
            SeaLionCoord(83, 2, 46, 4423),      # Empty sea?
            SeaLionCoord(259, 0, 1358, 2228),   # Empty sea (kudos: @authman) 
            SeaLionCoord(275, 0, 272, 4701),    # Empty sea (kudos: @authman) 
            SeaLionCoord(292, 2, 4, 248),       # Rock
            SeaLionCoord(303, 3, 1533, 3337),   # Rock
            SeaLionCoord(741, 0, 1418, 3258),   # Empty sea (kudos: @authman) 
            SeaLionCoord(741, 0, 2466, 3700),   # Empty sea (kudos: @authman) 
            SeaLionCoord(912, 2, 813, 3117),    # Random dot on tail of adult male
            SeaLionCoord(921, 3, 2307, 1418),   # Empty sea
            SeaLionCoord(921, 3, 2351, 1398),   # Empty sea
        )
        self.bad_coords = to_tid_coords(bad_coords)
        
        missing_coords = ( 
            SeaLionCoord(15, 2, 1686, 2620),    # Merged double dot, both rejected
            SeaLionCoord(148, 1, 1390, 4525),
            SeaLionCoord(607, 0, 1160, 2459),
            SeaLionCoord(607, 0, 1245, 2836),
            SeaLionCoord(816, 3, 2256, 767),
            SeaLionCoord(899, 2, 550, 2114),    # adult_female or juvenile?
        )
        self.missing_coords = to_tid_coords(missing_coords)


        # Corrections to train.csv counts 
        self.better_counts = {
            2 : (2, 0, 37, 19, 0),      # (kudos: @authman)
            11 : (3, 5, 36, 13, 0),     # (kudos: @authman)
            13 : (1, 5, 20, 13, 0),
            15 : (2, 3, 33, 56, 0),
            18 : (2, 3, 0, 0, 0), 
            36 : (8, 17, 0, 0, 0),
            38 : (3, 0, 33, 0, 0),
            40 : (2, 2, 62, 7, 0),      # (kudos: @authman)
            47 : (13, 14, 48, 3, 33),
            52 : (2, 3, 20, 23, 0),
            66 : (8, 5, 23, 17, 2),     # train.csv reports no sea lions, but lots annotated
            83 : (5, 2, 44, 41, 0),
            148: (0, 3, 0, 5, 0),
            221: (6, 1, 26, 9, 2),      # (kudos: @authman)
            292: (5, 5, 49, 42, 1), 
            299: (27, 9, 209, 32, 55),  # (kudos: @authman)
            312: (1, 1, 21, 14, 0),     # (kudos: @authman)
            335: (6, 36, 18, 12, 0),    # (kudos: @authman)
            426: (2, 6, 11, 42, 5),     # (kudos: @authman)
            479: (5, 4, 0, 0, 0),       # (kudos: @authman)
            492: (2, 1, 9, 21, 1),      # (kudos: @authman)
            510: (5, 1, 0, 0, 0),       # (kudos: @authman)
            529: (5, 2, 15, 12, 0),     # (kudos: @authman)
            538: (10, 2, 162, 9, 115),  # (kudos: @authman)
            577: (3, 1, 116, 97, 0),    # (kudos: @authman)
            593: (1, 2, 32, 58, 0),     # (kudos: @authman)       
            607: (2, 3, 14, 3, 0),
            643: (1, 5, 0, 21, 0),
            698: (1, 0, 0, 14, 0),      # (kudos: @authman)
            706: (2, 4, 39, 18, 0),     # (kudos: @authman)
            707: (4, 21, 1, 7, 0),      # (kudos: @authman)
            776: (8, 2, 25, 2, 29),     # (kudos: @authman)
            899: (2, 2, 4, 3, 0),       # (kudos: @authman)
            912: (30, 2, 247, 13, 205),
        }
        
        # train_ids that arn't in bad_train_ids but still have some discrepancy 
        # with train.csv counts
        self.anomalous_train_ids = (
            62, 63, 67, 73, 77, 78, 80, 87, 91, 93,
            99, 105, 108, 110, 122, 127, 134, 136, 146, 155,
            170, 174, 175, 177, 178, 179, 181, 186, 187, 207,
            211, 214, 216, 218, 240, 252, 256, 258, 265, 271,
            277, 292, 293, 297, 298, 309, 310, 323, 325, 328,
            330, 338, 342, 351, 359, 361, 362, 365, 368, 369,
            375, 382, 383, 386, 388, 394, 395, 398, 405, 409,
            410, 412, 416, 418, 431, 433, 437, 441, 460, 462,
            465, 467, 473, 475, 476, 482, 483, 487, 495, 498,
            500, 505, 509, 516, 518, 523, 524, 539, 543, 544, 
            545, 552, 553, 554, 555, 568, 571, 574, 578, 585, 
            587, 595, 598, 604, 606, 619, 629, 632, 633, 645, 
            655, 658, 662, 664, 668, 675, 676, 679, 686, 699, 
            700, 703, 710, 724, 729, 732, 739, 744, 745, 748, 
            750, 751, 754, 759, 761, 763, 764, 781, 788, 790, 
            795, 798, 803, 804, 805, 806, 813, 814, 822, 823, 
            827, 837, 845, 858, 865, 871, 873, 878, 881, 882, 
            889, 900, 906, 910, 914, 917, 918, 920, 921, 924, 
            925, 926, 933, 934, 937)
        
        # caches
        self._tid_counts = None
        self._tid_coords = None

            
    
    @property
    def trainshort1_ids(self):
        tids = range(0, 11)
        tids = self._remove_bad_ids(tids)
        return tids


    @property
    def trainshort2_ids(self):
        tids = range(41,51)
        tids = self._remove_bad_ids(tids)
        return tids

        
    @property 
    def train_ids(self):
        """List of all valid train ids"""
        tids = range(0, self.train_nb)
        tids = self._remove_bad_ids(tids)
        return tids


    def _remove_bad_ids(self, tids) :
        tids = list(set(tids) - set(self.bad_train_ids) )
        tids.sort()
        return tids
        
        
    @property 
    def test_ids(self):
        return range(0, self.test_nb)
 
    
    def path(self, name, **kwargs):
        """Return path to various source files"""
        path = self.paths[name].format(**kwargs)
        return path        


    @property
    def tid_counts(self) :
        """A map from train_id to list of sea lion class counts"""
        if self._tid_counts is None :
            tid_counts = OrderedDict()
            fn = self.path('counts')
            with open(fn) as f:
                f.readline()
                for line in f:
                    counts = tuple(map(int, line.split(',')))
                    tid_counts[counts[0]] = counts[1:]
            # Apply corrections
            for tid, counts in self.better_counts.items() :
                tid_counts[tid] = counts
            self._tid_counts = tid_counts
        return self._tid_counts


    def count_coords(self, tid_coords) :
        """Return a map from ids to list of class counts.
        
        Args: 
            tid_coords: A map from ids to coordinates.
        
        Returns:
            A list of integer sea lion class counts 
        
        """
        tid_counts = OrderedDict()
        for tid, coords in tid_coords.items(): 
            counts = [0]*self.cls_nb
            for tid, cls, row, col in coords :
                counts[cls] +=1
            tid_counts[tid] = counts
        return tid_counts


    def rmse(self, tid_counts) :        
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
        error_fraction = err_nb / (len(tid_counts)* self.cls_nb )
         
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
        if mask :
            # The masked areas are not uniformly black, presumable due to 
            # jpeg compression artifacts
            MASK_MAX = 40 
            dot_img = self.load_dotted_image(train_id, scale, border).astype(np.uint16).sum(axis=-1)
            img = np.copy(img)
            img[dot_img<MASK_MAX] = 0 
        return img
   

    def load_dotted_image(self, train_id, scale=1, border=0):
        img = self._load_image('dotted', train_id, scale, border)
        
        # When dotted image is rotated relative to train, apply hot patch. (kudos: @authman)
        if train_id in self.dotted_rotate :
            rot = self.dotted_rotate[train_id]
            img = np.rot90(img, rot)
            
        return img
 
 
    def load_test_image(self, test_id, scale=1, border=0):    
        return self._load_image('test', test_id, scale, border)


    def _load_image(self, itype, tid, scale=1, border=0) :
        fn = self.path(itype, tid=tid)
        
        # Workaround for weird issie in pillow that throws ResourceWarnings
        with open(fn, 'rb') as img_file: 
            with Image.open(img_file) as image:
                if scale != 1 :
                    width, height  = image.size # width x height for PIL
                    image = image.resize((width//scale, height//scale), Image.ANTIALIAS)
        
                img = np.asarray(image)
        
        if border :
            height, width, channels = img.shape
            bimg = np.zeros( shape=(height+border*2, width+border*2, channels), dtype=np.uint8)
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
        # (But if we set this as defualt we pick up extra spurious dots elsewhere)
        if train_id in [491, 816] : MAX_COLOR_DIFF = 48
       
        src_img = np.asarray(self.load_train_image(train_id, mask=True), dtype = np.float)
        dot_img = np.asarray(self.load_dotted_image(train_id), dtype = np.float)

        # Sometimes the exposure of the Train and TrainDotted images is different. 
        # If mismatch is not too bad we can sometimes fix this problem.
        # (see also comments on bad_train_ids)
        ratio = src_img.sum() / dot_img.sum()
        MISMATCHED_EXPOSURE = 1.05
        if ratio > MISMATCHED_EXPOSURE or ratio < 1/MISMATCHED_EXPOSURE:
            self._progress(' (Adjusting exposure: {} {})'.format(train_id, ratio), verbosity=VERBOSITY.VERBOSE)
            # We adjust the source image so not to mess up dot colors
            src_img /= ratio
            
        img_diff = np.abs(src_img - dot_img)
        
        # Detect bad data. If train and dotted images are very different then somethings wrong.
        avg_diff = img_diff.sum() / (img_diff.shape[0] * img_diff.shape[1])
        if avg_diff > MAX_AVG_DIFF: 
            self._progress('( Bad train image -- exceeds MAX_AVG_DIFF: {} )'.format(train_id))
            return ()
        
        img_diff = np.max(img_diff, axis=-1)   
           
        img_diff[img_diff<MIN_DIFFERENCE] = 0
        img_diff[img_diff>=MIN_DIFFERENCE] = 255

        sealions = []
        
        for cls, color in enumerate(self.cls_colors):
            # color search backported from @bitsofbits.
            # Note that there are large red boxes and arrows in some training images (e.g. 912)
            # The red of these lines (250,0,10) is sufficiently different from dot red that the
            # lines mostly get filtered out.
            color_array = np.array(color)[None, None, :]
            has_color = np.sqrt(np.sum(np.square(dot_img * (img_diff > 0)[:,:,None] - color_array), axis=-1)) < MAX_COLOR_DIFF 
            contours = skimage.measure.find_contours(has_color.astype(float), 0.5)
            
            if self.verbosity == VERBOSITY.DEBUG :
                print()
                fn = 'diff_{}_{}.png'.format(train_id,cls)
                print('Saving train/dotted difference: {}'.format(fn))
                Image.fromarray((has_color*255).astype(np.uint8)).save(fn)

            for cnt in contours :
                p = Polygon(shell=cnt)
                area = p.area 
                if(area > MIN_AREA and area < MAX_AREA) :
                    row, col = p.centroid.coords[0] # DANGER : skimage and cv2 coordinates transposed
                    row = int(round(row))
                    col = int(round(col))
                    
                    NO_RED_DOT_CORRECTION = [6,]
                    if cls == self.cls_idx.adult_males and train_id not in NO_RED_DOT_CORRECTION:
                        # Sometimes there are ends of red lines poking out from under the black masks
                        # that get mistaken for adult male red dots.
                        dot_region = src_img[row-4: row+5, col-4:col+5]
                        zero_count = dot_region.size - np.count_nonzero(dot_region)  
                        
                        if zero_count>MAX_MASK:
                            self._progress(' (Rejecting {} 0 {} {} : {})'.format(train_id, row, col, zero_count),
                                verbosity=VERBOSITY.DEBUG)
                            continue

                    # Remove known bad coordinates
                    if train_id in self.bad_coords :
                        bad_coords = self.bad_coords[train_id]
                        if any([c.cls==cls and abs(c.row-row)<2 and abs(c.col-col)<2 for c in bad_coords]) :
                            self._progress(' (Removing bad coord: {} {} {} {})'.format(train_id, cls, row, col),
                                verbosity=VERBOSITY.DEBUG)
                            continue
                    #print(train_id, cls, row, col, dot_img[row, col])

                    sealions.append( SeaLionCoord(train_id, cls, row, col) )
        
        if train_id in self.missing_coords :
            sealions.extend(self.missing_coords[train_id])  
        
        if self.verbosity >= VERBOSITY.VERBOSE :
            counts = [0,0,0,0,0]
            for c in sealions :
                counts[c.cls] +=1
            print()
            print('train_id','true_counts','counted_dots', 'difference', sep='\t')   
            true_counts = self.tid_counts[train_id]
            print(train_id, true_counts, counts, np.array(true_counts) - np.array(counts) , sep='\t' )
          
        if self.verbosity == VERBOSITY.DEBUG :
            img = np.copy(self.load_dotted_image(train_id))
            delta = self.dot_radius
            for tid, cls, row, col in sealions :                    
                for r in range(row-delta, row+delta+1) : img[r, col, :] = 255
                for c in range(col-delta, col+delta+1) : img[row, c, :] = 255    
            fn = 'cross_{}.png'.format(train_id)
            print('Saving crossed dots: {}'.format(fn))
            Image.fromarray(img).save(fn)
     
        return sealions
        

    def save_coords(self, train_ids=None): 
        if train_ids is None: train_ids = self.train_ids 
        fn = self.path('coords')
        self._progress('Saving sea lion coordinates to {}'.format(fn))
        if os.path.exists(fn) :
            raise IOError('Output file exists: {}'.format(fn))
        
        # Multiprocessing support (Kudos: @JandJ)
        if MULTIPROCESSING :
            all_coord_list = Pool().map(self.find_coords, train_ids)
        else :
            all_coord_list = map(self.find_coords, train_ids)
            
        with open(fn, 'w') as csvfile:
            writer =csv.writer(csvfile)
            writer.writerow( SeaLionCoord._fields )
            for coord in all_coord_list:
                for c in coord:
                    writer.writerow(c)
        self._progress('done')

    
    @property
    def tid_coords(self):
        """Loads the coordinates saved by save_coords() and return a dictionary from tid to SeaLionCoords"""
        if self._tid_coords is None :
            fn = self.path('coords')
            if not os.path.exists(fn) : self.save_coords()
            
            self._progress('( Loading sea lion coordinates from {}'.format(fn))
            with open(fn) as f:
                f.readline()
                slc = [SeaLionCoord(*[int(n) for n in line.split(',')]) for line in f]
            self._progress(')')  
            #tid_coords = OrderedDict()
            #for c in slc :
            #    tid = c.tid
            #    if tid not in tid_coords: tid_coords[tid] = []
            #    tid_coords[tid].append(c)
            self._tid_coords = to_tid_coords(slc)
        return self._tid_coords     

            
    def save_sea_lions(self, train_id, coords, size=TILE_SIZE, dotted=False):
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
        
        if dotted :
            img = self.load_dotted_image(train_id, border=size//2)
        else :
            img = self.load_train_image(train_id, border=size//2, mask=True)
            
        for tid, cls, row, col in coords :
            assert(tid==train_id)
            fn = self.path('chunk', size=size, tid=tid, cls=cls, row=row, col=col)
            self._progress(' Saving '+fn, end='\n', verbosity=VERBOSITY.VERBOSE)
            Image.fromarray( img[row:row+size, col:col+size, :]).save(fn)
            self._progress()
            
        self._progress('done')
        
            
    def _progress(self, string=None, end=' ', verbosity=VERBOSITY.NORMAL):
        if self.verbosity < verbosity: return
        if not string :
            print('.', end='')
        elif string == 'done':
            print(' done') 
        else:
            print(string, end=end)
        sys.stdout.flush()

# end SeaLionData

# Utility routines
def dump_namedtuple(filename, tuple_type, list_of_namedtuples) :
    with open(filename, 'w') as csvfile:
        writer =csv.writer(csvfile)
        writer.writerow(tuple_type._fields )
        for item in list_of_namedtuples :
            writer.writerow(item)


# Round up to next size
def roundup(x, size):
    return ((x+size-1) // size) * size


# Round down to previous size
def rounddown(x, size):
    return roundup(x-size+1, size)


def to_tid_coords(coords) :
    """Convert list of SeaLionCoords to map for tid to lists of coords"""
    tid_coords = OrderedDict()
    for c in coords :
        tid = c.tid
        if tid not in tid_coords: 
            tid_coords[tid] = []
        tid_coords[tid].append(c)
    return tid_coords
               


if __name__ == "__main__":
    # Build coordinates
    sld = SeaLionData()
    sld.save_coords()

    # Error analysis
    sld.verbosity = VERBOSITY.VERBOSE
    tid_counts = sld.count_coords(sld.tid_coords)
    rmse, frac = sld.rmse(tid_counts)

    print()
    print('RMSE: {}'.format(rmse) )
    print('Error frac: {}'.format(frac))
   
    
    