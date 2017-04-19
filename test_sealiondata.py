#!/usr/bin/env python

import unittest

from sealiondata import *



class TestUtils(unittest.TestCase):
    def test_round(self):
        self.assertEqual(100, roundup(94,10) )
        self.assertEqual(90, rounddown(94,10) )
        self.assertEqual(100, roundup(100,10) )
        self.assertEqual(100, rounddown(100,10) )


class TestSeaLionData(unittest.TestCase):
    def setUp(self):
        self.sld = SeaLionData()
        
    def tearDown(self):
        pass

    def test_data(self):
        # TODO
        pass
    
    def test_ids(self) :
        self.assertIsNotNone(self.sld.trainshort1_ids)
        self.assertIsNotNone(self.sld.trainshort2_ids)
        self.assertIsNotNone(self.sld.train_ids)
        self.assertIsNotNone(self.sld.test_ids)


    def test_path(self) :
        fn = self.sld.path('sample')
        self.assertEqual( fn, os.path.join('..', 'input', 'sample_submission.csv'))
        
        fn = self.sld.path('train', tid=2)
        self.assertEqual( fn, os.path.join('..', 'input', 'Train', '2.jpg'))

    def test_tid_counts(self):
        counts = self.sld.tid_counts[899]
        self.assertEqual(counts, self.sld.better_counts[899])

    def test_count_coords(self):
        coords = self.sld.tid_coords[42]
        counts = self.sld.count_coords({42:coords})[42]
        true_counts = self.sld.tid_counts[42]
        
        self.assertTrue( any([a==b for a,b in zip(counts, true_counts)]) )

    def test_rmse(self):
        counts = list(self.sld.tid_counts[42])
        counts = [ c+1 for c in counts]
        rmse, err_frac = self.sld.rmse({42:counts}) 
        self.assertEqual(rmse, 1.0)
        self.assertEqual(err_frac, 1.0)


    def test_load_train_image(self) :
        img = self.sld.load_train_image(42)
        self.assertEqual((3328, 4992, 3), img.shape)

        img = self.sld.load_train_image(42, scale=10)
        self.assertEqual((332, 499, 3), img.shape)
        
        img = self.sld.load_train_image(42, border=1)
        self.assertEqual((3330, 4994, 3), img.shape)
        
        # TODO: Test mask copy


    def test_load_dotted_image(self):
        img = self.sld.load_dotted_image(7) # hot patched
        self.assertEqual((3744, 5616, 3), img.shape)
        
        
    def test_load_test_image(self):
        # TODO
        pass
        
    def test_find_coords(self):
        coords = self.sld.find_coords(66)
        counts = self.sld.count_coords({66:coords})[66]
        true_counts = self.sld.better_counts[66]
        self.assertTrue( any([a==b for a,b in zip(counts, true_counts)]) )
        
    
    def test_tid_coords(self):
        coords = self.sld.tid_coords[22]
        self.assertEqual(len(coords),7)

    
    
    

if __name__ == '__main__':
    unittest.main()
