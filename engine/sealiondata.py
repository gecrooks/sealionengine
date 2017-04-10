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
# row, col, ch -- Images are indexed as (rows, columns, channels) with origin at top left. (skimage conventions)
#             Beware: Some libraries use (x,y) cartesian coordinates (e.g. cv2, matplotlib)
# rr, cc -- lists of row and column coordinates (another skimage convention)
#
# With contributions from @bitsofbits ...
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

OUTDIR = '.'    #FIXME

VERBOSITY = namedtuple('VERBOSITY', ['QUITE', 'NORMAL', 'VERBOSE', 'DEBUG'])(0,1,2,3)

SeaLionCoord = namedtuple('SeaLionCoord', ['tid', 'cls', 'row', 'col'])

TileCoord = namedtuple('TileCoord', ['tid', 'row', 'row_stop', 'col', 'col_stop'])

TILE_SIZE = 128

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
            'chunk'  : os.path.join(outdir, 'chunk_{tid}_{cls}_{row}_{col}_{size}.png'),
            }
        

        self.bad_train_ids = (
            # From MismatchedTrainImages.txt
            3, 7, 9, 21, 30, 34, 71, 81, 89, 97, 151, 184, 215, 234, 242, 
            268, 290, 311, 331, 344, 380, 384, 406, 421, 469, 475, 490, 499, 
            507, 530, 531, 605, 607, 614, 621, 638, 644, 687, 712, 721, 767, 
            779, 781, 794, 800, 811, 839, 840, 869, 882, 901, 903, 905, 909, 
            913, 927, 946,
            # Additional
            857,    # Many sea lions, but no dots on sea lions
            )
            
        self._counts = None

        
    @property
    def trainshort_ids(self):
        return (0,1,2,4,5,6,8,10)  # Trainshort1
        #return range(41,51)        # Trainshort2
        
    @property 
    def train_ids(self):
        """List of all valid train ids"""
        tids = range(0, self.train_nb)
        tids = list(set(tids) - set(self.bad_train_ids) )  # Remove bad ids
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
    def counts(self) :
        """A map from train_id to list of sea lion class counts"""
        if self._counts is None :
            counts = {}
            fn = self.path('counts')
            with open(fn) as f:
                f.readline()
                for line in f:
                    tid_counts = list(map(int, line.split(',')))
                    counts[tid_counts[0]] = tid_counts[1:]
            self._counts = counts
        return self._counts

    def rmse(self, tid_counts) :
        true_counts = self.counts
        
        error = np.zeros(shape=[5] )
        
        for tid in tid_counts:
            true_counts = self.counts[tid]
            obs_counts = tid_counts[tid]
            diff = np.asarray(true_counts) - np.asarray(obs_counts)
            error += diff*diff
        #print(error)
        error /= len(tid_counts)
        rmse = np.sqrt(error).sum() / 5
        return rmse 
        

    def load_train_image(self, train_id, border=0, mask=False):
        """Return image as numpy array
         
        border -- add a black border of this width around image
        mask -- If true mask out masked areas from corresponding dotted image
        """
        img = self._load_image('train', train_id, border)
        if mask :
            # The masked areas are not uniformly black, presumable due to 
            # jpeg compression artifacts
            dot_img = self._load_image('dotted', train_id, border).astype(np.uint16).sum(axis=-1)
            img = np.copy(img)
            img[dot_img<40] = 0     #FIXME magic constant
        return img
   

    def load_dotted_image(self, train_id, border=0):
        return self._load_image('dotted', train_id, border)
 
 
    def load_test_image(self, test_id, border=0):    
        return self._load_image('test', test_id, border)


    def _load_image(self, itype, tid, border=0) :
        fn = self.path(itype, tid=tid)
        img = np.asarray(Image.open(fn))
        if border :
            height, width, channels = img.shape
            bimg = np.zeros( shape=(height+border*2, width+border*2, channels), dtype=np.uint8)
            bimg[border:-border, border:-border, :] = img
            img = bimg
        return img
    

    def coords(self, train_id):
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
        
        # Detect bad data. If train and dotted images are very different then somethings wrong.
        avg_diff = img_diff.sum() / (img_diff.shape[0] * img_diff.shape[1])
        if avg_diff > MAX_AVG_DIFF: return None
        
        img_diff = np.max(img_diff, axis=-1)   
           
        img_diff[img_diff<MIN_DIFFERENCE] = 0
        img_diff[img_diff>=MIN_DIFFERENCE] = 255

        sealions = []
        
        for cls, color in enumerate(self.cls_colors):
            # color search backported from @bitsofbits.
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
            true_counts = self.counts[train_id]
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
        if train_ids is None: train_ids = self.trainshort_ids #self.train_ids #FIXME
        fn = self.path('coords')
        self._progress('Saving sea lion coordinates to {}'.format(fn))
        with open(fn, 'w') as csvfile:
            writer =csv.writer(csvfile)
            writer.writerow( SeaLionCoord._fields )
            for tid in train_ids :
                self._progress()
                for coord in self.coords(tid):
                    writer.writerow(coord)
        self._progress('done')
    
    def load_coords(self):
        fn = self.path('coords')
        self._progress('Loading sea lion coordinates from {}'.format(fn))
        with open(fn) as f:
            f.readline()
            return [SeaLionCoord(*[int(n) for n in line.split(',')]) for line in f]
                           
            
    def save_sea_lions(self, coords=None, size=TILE_SIZE, dotted=False):
        """Save image chunks of given size centered on sea lion coordinates.
        If no coordinates given, then load training set coordinates.
        """
        self._progress('Saving image chunks...')
        self._progress('\n', verbosity=VERBOSITY.VERBOSE)
        
        if coords is None : coords = self.load_coords()
        
        last_tid = -1
        
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



def main():
    """SeaLionData command line interface"""
    
    parser = argparse.ArgumentParser(
        description=__description__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--version', action='version', version=__version__)
    parser.add_argument('-S', '--sourcedir', action='store', dest='sourcedir',
                        default=SOURCEDIR,  metavar='PATH',
                        help='Location of source data')
    parser.add_argument('-O', '--outdir', action='store', dest='outdir',
                        default=OUTDIR,  metavar='PATH',
                        help='Location of processed data')                        
    parser.add_argument('-V', '--verbosity', action='store', dest='verbosity',
                        default=VERBOSITY.NORMAL,  metavar='N', type=int,
                        help='Verbosity level (0:quite, 1:normal, 2:verbose, 3:debug')  


    cmdparser = parser.add_subparsers(title='Commands', description=None,
                                      help="-h, --help Additional help",)
    
    cmd = cmdparser.add_parser('save_coords', help='Build bands and targets.')
    cmd.set_defaults(funcname='save_coords')

    cmd = cmdparser.add_parser('save_sea_lions', help='Save collection of sealion images')
    cmd.set_defaults(funcname='save_sea_lions')
    

    opts = vars(parser.parse_args())

    sourcedir = opts.pop('sourcedir')
    outdir = opts.pop('outdir')
    verbosity = opts.pop('verbosity')
    funcname = opts.pop('funcname')

        
    sld = SeaLionData(sourcedir=sourcedir, outdir=outdir,verbosity=verbosity)
    func = getattr(sld, funcname)
    func(**opts)
    
       
       
if __name__ == "__main__":
    main()
    

