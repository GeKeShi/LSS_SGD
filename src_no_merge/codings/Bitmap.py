#!/usr/bin/env python
# coding: utf-8

import numpy as np

class BitMap(object):


    def __init__(self):
        """
        Create a BitMap
        """
        self.bm = None
        

    def __del__(self):
        """
        Destroy the BitMap
        """
        pass

    def pack(self, cluster_array):
        """
        Set the value of bit@pos to 1
        """
        # self.bitmap[pos // 8] |= self.BITMASK[pos % 8]
        self.bm = np.packbits(cluster_array, axis=-1)


    def reset(self, pos):
        """
        Reset the value of bit@pos to 0
        """
        # self.bitmap[pos // 8] &= ~self.BITMASK[pos % 8]
        pass


    def size(self):
        """
        Return size
        """
        return len(self.bm)

    @staticmethod
    def merge_bitmap(bmlist):
        merged_bitmap = np.zeros((bmlist[0].unpack()).shape, dtype=np.uint8)
        # print(args)
        for bitmap in bmlist:
            # np.add(merged_bitmap, np.unpackbits(bitmap, axis=-1), out=merged_bitmap)
            print(bitmap.size())
            np.add(merged_bitmap, bitmap.unpack(), out=merged_bitmap)
        return merged_bitmap

    def unpack(self):
        return np.unpackbits(self.bm, axis=-1)


    def query(self, unpacked_bitmap, index):
        return unpacked_bitmap[index]
        
        


