#!/usr/bin/env python

"""Sea Lion Prognostication Engine

https://www.kaggle.com/c/noaa-fisheries-steller-sea-lion-population-count
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

# Notes
# cls -- sea lion class 
# tid -- train, train dotted, or test image id 
# _nb -- abbreviation for number
#
# row, col, ch -- Image arrays are indexed as (rows, columns, channels) with origin at top left. 
#             Beware: Some libraries use (x,y) cartesian coordinates (e.g. cv2, matplotlib)
# rr, cc -- lists of row and column coordinates 
#
# By default, SeaLionData expects source data to be located in ../input, and saves processed data to ./outdir
#
#
# With contributions from @bitsofbits, @authman, @mfab ...
#


# ================ Meta ====================
__description__ = 'Sea Lion Prognostication Engine'
__version__ = '0.1.0'
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
            (87,46,10),         # brown 
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
            3, 
            # 7,    # TrainDotted rotated 180 degrees. Apply custom fix in load_dotted_image()
            9, 21, 30, 34, 71, 81, 89, 97, 151, 184, 215, 234, 242, 
            268, 290, 311, 331, 344, 380, 384, 406, 421, 469, 475, 490, 499, 
            507, 530, 531, 605, 607, 614, 621, 638, 644, 687, 712, 721, 767, 
            779, 781, 794, 800, 811, 839, 840, 869, 882, 901, 903, 905, 909, 
            913, 927, 946,
            # Additional
            857,    # Many sea lions, but no dots on sea lions
            )
            
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
                    counts = list(map(int, line.split(',')))
                    tid_counts[counts[0]] = counts[1:]
            self._tid_counts = tid_counts
        return self._tid_counts


    def count_coords(self, tid_coords) :
        """Take a map from ids to coordinates, 
        and return a map from ids to list of class counts"""
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
         
        border -- add a black border of this width around image
        mask -- If true copy masks from corresponding dotted image
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
        
        if train_id == 7 :
            # dotted image is rotated relative to train. Apply custom fix. (kudos: @authman)
            img = np.rot90(img, 2, (0,1) )
            
        return img
 
 
    def load_test_image(self, test_id, scale=1, border=0):    
        return self._load_image('test', test_id, scale, border)


    def _load_image(self, itype, tid, scale=1, border=0) :
        fn = self.path(itype, tid=tid)
        img = Image.open(fn) 
        
        if scale != 1 :
            width, height  = img.size # width x height for PIL
            img = img.resize((width//scale, height//scale), Image.ANTIALIAS)
        
        img = np.asarray(img)
        
        if border :
            height, width, channels = img.shape
            bimg = np.zeros( shape=(height+border*2, width+border*2, channels), dtype=np.uint8)
            bimg[border:-border, border:-border, :] = img
            img = bimg
        return img
    

    def find_coords(self, train_id):
        """Extract coordinates of dotted sealions and return list of SeaLionCoord objects"""
        
        # Empirical constants
        MIN_DIFFERENCE = 16
        MIN_AREA = 9
        MAX_AREA = 100
        MAX_AVG_DIFF = 50
        MAX_COLOR_DIFF = 32
       
        src_img = np.asarray(self.load_train_image(train_id, mask=True), dtype = np.float)
        dot_img = np.asarray(self.load_dotted_image(train_id), dtype = np.float)

        img_diff = np.abs(src_img-dot_img)
        
        #print(src_img.shape, dot_img.shape)
        #Image.fromarray( src_img.astype(np.uint8) ).save('src_img.png')
        #Image.fromarray( dot_img.astype(np.uint8) ).save('dot_img.png')
        #sys.exit()
        
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
            # The red of these lines (250,0,10) is sufficiently different from dot red that the lines get filtered out.
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
                    sealions.append( SeaLionCoord(train_id, cls, row, col) )
                
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
        with open(fn, 'w') as csvfile:
            writer =csv.writer(csvfile)
            writer.writerow( SeaLionCoord._fields )
            for tid in train_ids :
                self._progress()
                for coord in self.find_coords(tid):
                    writer.writerow(coord)
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
            tid_coords = OrderedDict()
            for c in slc :
                tid = c.tid
                if tid not in tid_coords: tid_coords[tid] = []
                tid_coords[tid].append(c)
            self._tid_coords = tid_coords
        return self._tid_coords     

            
    def save_sea_lions(self, coords=None, size=TILE_SIZE, dotted=False):
        """Save image chunks of given size centered on sea lion coordinates.
        If no coordinates given, then load training set coordinates.
        """
        self._progress('Saving image chunks...')
        self._progress('\n', verbosity=VERBOSITY.VERBOSE)
        
        if coords is None : coords = self.load_coords()
        
        last_tid = None
        
        for tid, cls, row, col in coords :
            if tid != last_tid:
                if dotted :
                    img = self.load_dotted_image(tid, border=size//2)
                else :
                    img = self.load_train_image(tid, border=size//2, mask=True)
                last_tid = tid

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
   
    
    