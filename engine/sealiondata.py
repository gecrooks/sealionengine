#!/usr/bin/env python

"""Sea Lion Prognostication Engine

https://www.kaggle.com/c/noaa-fisheries-steller-sea-lion-population-count
"""

import sys
import os
from collections import namedtuple
import operator
import glob
import csv 
from math import sqrt

import numpy as np

from PIL import Image, ImageDraw, ImageFilter

import cv2

import shapely
import shapely.geometry
from shapely.geometry import Polygon



# Notes
# tid : train or test image id 
# x,y : don't forget image arrays organized row,col,channels

# ================ Meta ====================
__description__ = 'Sea Lion Prognostication Engine'
__version__ = '0.1.0'
__license__ = 'MIT'
__author__ = 'Gavin Crooks (@threeplusone)'
__status__ = "Prototype"
__copyright__ = "Copyright 2017"

# python -c 'import sealionengine.py; sealionengine.package_versions()'
def package_versions():
    print('sealionengine        \t', __version__)
    print('python      \t', sys.version[0:5])
    print('numpy       \t', np.__version__)
    print('openCV (cv2)\t', cv2.__version__)
    print('pillow (PIL)\t', PIL.__version__)
    print('shapely     \t', shapely.__version__)


SOURCEDIR = os.path.join('..', 'input')

DATADIR = '.'

VERBOSITY = namedtuple('VERBOSITY', ['QUITE', 'NORMAL', 'VERBOSE', 'DEBUG'])(0,1,2,3)


SeaLionCoord = namedtuple('SeaLionCoord', ['tid', 'cls', 'x', 'y'])


class SeaLionData(object):
    
    def __init__(self, sourcedir=SOURCEDIR, datadir=DATADIR, verbosity=VERBOSITY.NORMAL):
        self.sourcedir = sourcedir
        self.datadir = datadir
        self.verbosity = verbosity
        
        self.class_nb = 5
        
        self.class_names = (
            'adult_males',
            'subadult_males',
            'adult_females',
            'juveniles',
            'pups')
        
        self.class_colors = (
            (246,0,0),          # red
            (250,10,250),       # magenta
            (84,42,10),          # brown 
            (30,60,180),        # blue
            (40,180,20),        # green
            )
        
        self.train_nb = 947
        
        self.test_nb = 18636
       
        self.source_paths = {
            'sample'     : os.path.join(sourcedir, 'sample_submission.csv'),
            'counts'     : os.path.join(sourcedir, 'Train', 'train.csv'),
            'train'      : os.path.join(sourcedir, 'Train', '{tid}.jpg'),
            'dotted'     : os.path.join(sourcedir, 'TrainDotted', '{tid}.jpg'),
            'test'       : os.path.join(sourcedir, 'Test', '{tid}.jpg'),
            
            'coords'     : os.path.join(datadir, 'coords.csv'),  
            }
        
        # From MismatchedTrainImages.txt
        self.bad_train_ids = (
            3, 7, 9, 21, 30, 34, 71, 81, 89, 97, 151, 184, 215, 234, 242, 
            268, 290, 311, 331, 344, 380, 384, 406, 421, 469, 475, 490, 499, 
            507, 530, 531, 605, 607, 614, 621, 638, 644, 687, 712, 721, 767, 
            779, 781, 794, 800, 811, 839, 840, 869, 882, 901, 903, 905, 909, 
            913, 927, 946)
            
        self._counts = None

        
    @property
    def trainshort_ids(self):
        #return (0,1,2,4,5,6,8,10)  # Trainshort1
        return range(41,51)         # Trainshort2
        
    @property 
    def train_ids(self):
        tids = range(0, self.train_nb)
        tids = list(set(tids) - set(self.bad_train_ids) )  # Remove bad ids
        tids.sort()
        return tids
                    
    @property 
    def test_ids(self):
        return range(0, self.test_nb)
    
    def path(self, name, **kwargs):
        """Return path to various source files"""
        path = self.source_paths[name].format(**kwargs)
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
        print(error)
        error /= len(tid_counts)
        rmse = np.sqrt(error).sum() / 5
        return rmse 
        

    def load_train_image(self, train_id, border=0, mask=False):
        img = self._load_image('train', train_id, border)
        if mask :
            dot_img = self._load_image('dotted', train_id, border).astype(np.uint16).sum(axis=-1)
            img = np.copy(img)
            img[dot_img<40] = 0
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



    def class_from_color(self, rgb):
        """Guess sea lion class id from pixel color of sea lion dot"""
        MAX_DIFF = 60
        colors = np.array(self.class_colors) 
        diff = (abs(colors - rgb)).sum(axis=-1)
        cls = np.argmin(diff)
        
        if diff[cls]>MAX_DIFF: 
            #print(cls, rgb, self.class_colors[cls], diff[cls])
            return None
        
        return cls
            
    def coords(self, train_id):
        """Extract coordinates of dotted sealions and return list of SeaLionCoord objects)"""
        
        # Empirical constants
        MIN_DIFFERENCE = 30
        MIN_AREA = 12
        MAX_AREA = 100
        MAX_AVG_DIFF = 50
       
        src_img = np.asarray(self.load_train_image(train_id, mask=True), dtype = np.float)
        dot_img = np.asarray(self.load_dotted_image(train_id), dtype = np.float)

        img_diff = np.abs(src_img-dot_img)
        
        
        # Mask out black masks in dotted images 
        #img_diff[dot_img.sum(axis=-1) <50] = 0  #FIXME: Magic constant. Use mask option elsewhere            
        
        # Detect bad data. If train and dotted images are very different then somethings wrong.
        avg_diff = img_diff.sum() / (img_diff.shape[0] * img_diff.shape[1])
        if avg_diff > MAX_AVG_DIFF: return None
        
#        img_diff[dot_img.sum(axis=-1) <50] = 0              # Mask out black masks in dotted images
        img_diff = np.max(img_diff, axis=-1)
           
        img_diff[img_diff<MIN_DIFFERENCE] = 0
        img_diff[img_diff>=MIN_DIFFERENCE] = 255

        sealions = []

        contours = cv2.findContours(img_diff.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
        
        for cnt in contours :
            area = cv2.contourArea(cnt) 
            if(area>MIN_AREA and area<MAX_AREA) :
                p = Polygon(shell=cnt[:, 0, :])
                x,y = p.centroid.coords[0]
                x = int(round(x))
                y = int(round(y))
                cls = self.class_from_color(dot_img[y,x])
                if cls is None: continue
                # print(cls, x, y, dot_img[y,x])
                sealions.append( SeaLionCoord(train_id, cls, x, y) )
        
        sealions.sort(key = operator.attrgetter('y') )
        sealions.sort(key = operator.attrgetter('x') )
        sealions.sort(key = operator.attrgetter('cls') )
        
        if self.verbosity >= VERBOSITY.VERBOSE :
            counts = [0,0,0,0,0]
            for c in sealions :
                counts[c.cls] +=1
            print()
            print('train_id','true_counts','counted_dots', 'difference', sep='\t')   
            true_counts = self.counts[train_id]
            print(train_id, true_counts, counts, np.array(true_counts) - np.array(counts) , sep='\t' )
            
            
        if self.verbosity == VERBOSITY.DEBUG :
            print()
            fn = 'diff_{}.png'.format(train_id)
            print('Saving train/dotted difference: {}'.format(fn))
            Image.fromarray(img_diff.astype(np.uint8)).save(fn)
        
            img = np.copy(sld.load_dotted_image(train_id))
            for tid, cls, cx, cy in sealions :
                for x in range(cx-3, cx+4) : img[cy, x, :] = 255
                for y in range(cy-3, cy+4) : img[y, cx, :] = 255    
            fn = 'cross_{}.png'.format(train_id)
            print('Saving crossed dots: {}'.format(fn))
            Image.fromarray(img).save(fn)
     
        return sealions
        

    def save_coords(self, train_ids=None): 
        if train_ids is None: train_ids = self.train_ids
        fn = self.path('coords')
        self._progress('Saving sealion coordinates to {}'.format(fn))
        with open(fn, 'w') as csvfile:
            writer =csv.writer(csvfile)
            writer.writerow( SeaLionCoord._fields )
            for tid in train_ids :
                self._progress()
                for coord in self.coords(tid):
                    writer.writerow(coord)
        self._progress('done')
            
    def save_sea_lion_chunks(self, coords, chunksize=128):
        self._progress('Saving image chunks...')
        self._progress('\n', verbosity = VERBOSITY.VERBOSE)
        
        last_tid = -1
        
        for tid, cls, x, y in coords :
            if tid != last_tid:
                img = self.load_train_image(tid, border=chunksize//2, mask=True)
                last_tid = tid

            fn = 'chunk_{tid}_{cls}_{x}_{y}_{size}.png'.format(size=chunksize, tid=tid, cls=cls, x=x, y=y)
            self._progress(' Saving '+fn, end='\n', verbosity = VERBOSITY.VERBOSE)
            Image.fromarray( img[y:y+chunksize, x:x+chunksize, :]).save(fn)
            self._progress()
        self._progress('done')
        
            
    def _progress(self, string=None, end=' ', verbosity=VERBOSITY.NORMAL):
        if self.verbosity< verbosity : return
        if not string :
            print('.', end='')
        elif string == 'done':
            print(' done') 
        else:
            print(string, end=end)
        sys.stdout.flush()

# end SeaLionData


# Count sea lion dots and compare to truth from train.csv
sld = SeaLionData()

#for tid in sld.trainshort_ids :
#    img = sld.load_train_image(tid).sum(axis=-1)
#    print(img.min(), img.max() )


#Image.fromarray(sld.load_train_image(41, mask=True)) #.save('41m.png')
#sys.exit()

sld.verbosity = VERBOSITY.VERBOSE
tid_counts = {}
#for tid in sld.trainshort_ids :
for tid in (41,42):
    coord = sld.coords(tid)
    counts = [0,0,0,0,0]
    for c in coord :
        counts[c.cls] +=1
    tid_counts[tid] = counts
    
print(tid_counts)
rmse = sld.rmse(tid_counts)
print(rmse)
#    sld.save_sea_lion_chunks(coord)

     
        
        
        
        
        


