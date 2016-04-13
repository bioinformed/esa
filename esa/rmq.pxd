from __future__ import division

from libc.stdint cimport int8_t, int16_t, int32_t, int64_t
from libc.stdint cimport uint8_t, uint16_t, uint32_t, uint64_t
cimport numpy as np
cimport cython


import numpy as np
from numpy import log2, floor
from time import time


cpdef uint32_t rmq_minimum_size = 128
cdef uint32_t microblock_size = 8
cdef uint32_t block_size = 16
cdef uint32_t superblock_size = 256


cdef class RMQ_base(object):
    cpdef uint32_t query(self, uint32_t i, uint32_t j) except -1


cdef class RMQ_succinct(RMQ_base):
    cdef uint32_t microblock_count
    cdef uint32_t block_count
    cdef uint32_t superblock_count
    cdef np.ndarray a
    cdef np.ndarray microblock_type
    cdef np.ndarray precomputed_queries
    cdef np.ndarray blocks
    cdef np.ndarray superblocks


    cdef inline uint32_t microblock_num(self, uint32_t n)
    cdef inline uint32_t block_num(self, uint32_t n)
    cdef inline uint32_t superblock_num(self, uint32_t n)
    cpdef uint32_t query(self, uint32_t i, uint32_t j) except -1


cdef class RMQ_stupid(RMQ_base):
    cdef np.ndarray a
    cpdef uint32_t query(self, uint32_t i, uint32_t j) except -1


cpdef RMQ_base RMQ(np.ndarray[int32_t, ndim=1] lcp)
