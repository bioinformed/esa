from __future__ import division, print_function

from libc.stdint cimport int8_t, int16_t, int32_t, int64_t
from libc.stdint cimport uint8_t, uint16_t, uint32_t, uint64_t

from intrinsics cimport __builtin_clz, __builtin_ffs

cimport numpy as np
cimport cython


import numpy as np
from numpy import log2, floor


cpdef inline uint8_t log2i(uint32_t n):
    return 31 - __builtin_clz(n | 1)


cpdef inline int lsb(int32_t n):
    cdef int m = __builtin_ffs(n)
    return m - 1 if m else 0


cdef int8_t *high_bits = [~0, ~1, ~3, ~7, ~15, ~31, ~63, ~127]


@cython.boundscheck(False)
cpdef uint64_t catalan(uint32_t n):
    cdef np.ndarray[uint64_t, ndim=1] C = np.zeros(n, dtype=np.uint64)
    cdef uint32_t i

    C[0] = C[1] = 1
    for i in range(2, n):
        C[i] = ((4*i - 2) * C[i - 1]) // (i + 1)
    return C[-1]


@cython.boundscheck(False)
cpdef np.ndarray[uint64_t, ndim=2] catalan_triangle(uint32_t n):
    cdef np.ndarray[uint64_t, ndim=2] T = np.zeros((n, n), dtype=np.uint64)
    cdef uint32_t i, j

    for j in range(n):
        T[0, j] = 1

    for i in range(1, n):
        for j in range(i, n):
            T[i, j] = T[i - 1, j] + T[i, j - 1]

    return T


cdef inline uint32_t array_min2(int32_t *a, uint32_t i, uint32_t j):
    if a[i] <= a[j]:
        return i
    else:
        return j


cdef inline uint32_t array_min3(int32_t *a, uint32_t i, uint32_t j, uint32_t k):
    return array_min2(a, i, array_min2(a, j, k))


cdef uint32_t rmq_min_length = 128
cdef uint32_t microblock_size = 8
cdef uint32_t block_size = 16
cdef uint32_t superblock_size = 256


cdef class RMQ_base(object):
    cpdef uint32_t query(self, uint32_t i, uint32_t j) except -1:
        raise NotImplementedError()


cdef class RMQ_succinct(RMQ_base):
    # attributes defined in rmq.pxd

    cdef inline uint32_t microblock_num(self, uint32_t n):
        return n // microblock_size


    cdef inline uint32_t block_num(self, uint32_t n):
        return n // block_size


    cdef inline uint32_t superblock_num(self, uint32_t n):
        return n // superblock_size


    #@cython.boundscheck(False)
    def __init__(self, np.ndarray[int32_t, ndim=1] a):
        cdef uint32_t start, stop, i, j, p, q, m1, m2, g, t
        cdef uint32_t block_min, superblock_min, sb, pos
        cdef uint32_t n = len(a)
        self.a = a

        self.microblock_count = self.microblock_num(n - 1) + 1
        self.block_count = self.block_num(n - 1) + 1
        self.superblock_count = self.superblock_num(n - 1) + 1

        if n < self.block_count // (2 * block_size):
            raise ValueError('range too small for this implementation')

        cdef np.ndarray[uint64_t, ndim=2] cat = catalan_triangle(microblock_size + 1)

        # Precompute microblock queries

        cdef uint64_t c = cat[microblock_size, microblock_size]
        self.microblock_type = np.zeros(self.microblock_count, dtype=np.uint16)
        self.precomputed_queries = np.zeros((c, microblock_size), dtype=np.uint8)
        cdef uint8_t *pre_base = <uint8_t*>self.precomputed_queries.data
        cdef uint8_t *pre

        cdef np.ndarray[uint16_t, ndim=1] microblock_type = self.microblock_type
        cdef np.ndarray[uint8_t, ndim=2] precomputed_queries = self.precomputed_queries

        precomputed_queries[:, 0] = 1

        cdef int block_depth = int(floor(log2(superblock_size / block_size)))
        cdef int superblock_depth = int(floor(log2(self.superblock_count))) + 1

        self.blocks = np.zeros((block_depth, self.block_count), dtype=np.uint8)
        self.superblocks = np.zeros((superblock_depth, self.superblock_count), dtype=np.uint32)

        cdef np.ndarray[uint8_t, ndim=2] blocks = self.blocks
        cdef np.ndarray[uint32_t, ndim=2] superblocks = self.superblocks

        cdef np.ndarray[int32_t, ndim=1] gstack = np.zeros(microblock_size, dtype=np.int32)
        cdef uint32_t gstacksize
        cdef np.ndarray[int32_t, ndim=1] right_path = np.zeros(microblock_size + 1, dtype=np.int32)
        right_path[0] = np.iinfo(np.int32).min

        stop = 0
        for i in range(self.microblock_count):
            start, stop = stop, min(stop + microblock_size, n)

            q, p = microblock_size, microblock_size - 1
            t = 0
            right_path[1] = a[start]

            for pos in range(start + 1, stop):
                p -= 1
                while right_path[q - p - 1] > a[pos]:
                    t += cat[p, q]
                    q -= 1
                right_path[q - p] = a[pos]

            microblock_type[i] = t
            pre = pre_base + c*t

            if pre[0] == 1:
                pre[0] = 0
                gstacksize = 0
                for j in range(start, stop):
                    while gstacksize and a[j] < a[gstack[gstacksize - 1]]:
                        gstacksize -= 1
                    if gstacksize:
                        g = gstack[gstacksize - 1]
                        pre[j - start] = pre[g - start] | (1 << (g % microblock_size))
                    else:
                        pre[j - start] = 0
                    gstack[gstacksize] = j
                    gstacksize += 1

        # Init blocks and superblocks
        stop = superblock_min = sb = 0
        for i in range(self.block_count):
            start = block_min = stop
            stop = min(start + block_size, n)
            for pos in range(start, stop):
                if a[pos] < a[block_min]:
                    block_min = pos
                if a[pos] < a[superblock_min]:
                    superblock_min = pos
            blocks[0, i] = block_min - start
            if stop % superblock_size == 0 or stop == n:
                superblocks[0, sb] = superblock_min
                superblock_min = stop
                sb += 1

        # Fill blocks
        cdef int dist = 1
        for j in range(1, block_depth):
            for i in range(self.block_count - dist):
                m1 = blocks[j - 1, i] + i*block_size
                m2 = blocks[j - 1, i + dist] + (i + dist)*block_size
                if a[m1] <= a[m2]:
                    blocks[j, i] = blocks[j - 1, i]
                else:
                    blocks[j, i] = blocks[j - 1, i + dist] + dist*block_size
            for i in range(self.block_count - dist, self.block_count):
                blocks[j, i] = blocks[j - 1, i]
            dist *= 2

        # Fill superblocks
        dist = 1
        for j in range(1, superblock_depth):
            for i in range(self.superblock_count - dist):
                if a[superblocks[j - 1, i]] <= a[superblocks[j - 1, i + dist]]:
                    superblocks[j, i] = superblocks[j - 1, i]
                else:
                    superblocks[j, i] = superblocks[j - 1, i + dist]
            for i in range(self.superblock_count - dist, self.superblock_count):
                superblocks[j, i] = superblocks[j - 1, i]
            dist *= 2

    #@cython.boundscheck(False)
    cpdef uint32_t query(self, uint32_t i, uint32_t j) except -1:
        cdef uint32_t n = len(self.a)

        if j >= n or i > j:
            raise IndexError('invalid interval i={}, j={}'.format(i, j))

        if i == j:
            return i

        cdef uint32_t b, min_i, min_j, min_ij, t, k, o

        cdef int32_t *a = <int32_t*>self.a.data
        cdef uint16_t *mb_type = <uint16_t*>self.microblock_type.data
        cdef np.ndarray[uint8_t, ndim=2] pre = self.precomputed_queries
        cdef np.ndarray[uint8_t, ndim=2] blocks = self.blocks
        cdef np.ndarray[uint32_t, ndim=2] superblocks = self.superblocks

        cdef uint32_t mb_i = self.microblock_num(i)
        cdef uint32_t mb_j = self.microblock_num(j)

        cdef uint32_t mb_start_i = mb_i * microblock_size
        cdef uint32_t i_pos = i - mb_start_i

        # Query within same microblock
        if mb_i == mb_j:
            t = pre[mb_type[mb_i], j - mb_start_i] & high_bits[i_pos]
            return mb_start_i + lsb(t) if t else j

        cdef uint32_t mb_start_j = mb_j * microblock_size
        cdef uint32_t j_pos = j - mb_start_j

        # Compute min within microblocks
        t = pre[mb_type[mb_i], microblock_size - 1] & high_bits[i_pos]
        min_ij = mb_start_i + (lsb(t) if t else (microblock_size - 1))
        t = pre[mb_type[mb_j], j_pos]
        min_j = mb_start_j + lsb(t) if t else j
        min_ij = array_min2(a, min_ij, min_j)

        # Done if microblocks are adjacent
        if mb_j - mb_i < 2:
            return min_ij

        cdef uint32_t b_i = self.block_num(i)
        cdef uint32_t b_j = self.block_num(j)
        cdef uint32_t b_start_i = b_i * block_size
        cdef uint32_t b_start_j = b_j * block_size

        # Next microblock query
        if i < b_start_i + microblock_size:
            t = pre[mb_type[mb_i + 1], microblock_size - 1]
            min_i = (mb_start_i + microblock_size + lsb(t)) if t else (b_start_i + block_size - 1)
            min_ij = array_min2(a, min_ij, min_i)

        # Next microblock query
        if j >= b_start_j + microblock_size:
            t = pre[mb_type[mb_j - 1], microblock_size - 1]
            min_j = (b_start_j + lsb(t)) if t else (mb_start_j - 1)
            min_ij = array_min2(a, min_ij, min_j)

        cdef uint32_t block_diff = b_j - b_i

        if block_diff < 2:
            return min_ij

        # Perform out-of-block queries
        b_i += 1

        # Just one out of block query needed
        if b_start_j - b_start_i - block_size <= superblock_size:
            k = log2i(block_diff - 2)
            o = b_j - (1 << k)
            min_i = blocks[k, b_i] + b_i * block_size
            min_j = blocks[k, o] + o * block_size
            return array_min3(a, min_ij, min_i, min_j)

        # Superblock query needed
        cdef uint32_t sb_i = self.superblock_num(i)
        cdef uint32_t sb_j = self.superblock_num(j)

        b = self.block_num((sb_i + 1) * superblock_size)
        k = log2i(b - b_i)
        t = b + 1 - (1 << k)
        min_i = blocks[k, b_i] + b_i * block_size
        min_j = blocks[k, t] + t * block_size
        min_ij = array_min3(a, min_ij, min_i, min_j)

        b = self.block_num(sb_j * superblock_size)
        k = log2i(b_j - b)
        t = b_j - (1 << k)
        min_i = blocks[k, b-1] + (b-1)*block_size
        min_j = blocks[k, t] + t*block_size
        min_ij = array_min3(a, min_ij, min_i, min_j)

        # If superblocks are adjacent, then we're done
        cdef uint32_t sb_diff = sb_j - sb_i
        if sb_diff < 2:
            return min_ij

        # Otherwise, the superblock query
        k = log2i(sb_diff - 2)
        min_i = superblocks[k, sb_i + 1]
        min_j = superblocks[k, sb_j - (1 << k)]

        return array_min3(a, min_ij, min_i, min_j)

    property nbytes:
        def __get__(self):
            return (self.microblock_type.nbytes +
                    self.precomputed_queries.nbytes +
                    self.blocks.nbytes +
                    self.superblocks.nbytes)

    property nbits:
        def __get__(self):
            return self.nbytes * 8


    property overhead:
        def __get__(self):
            return self.nbits / len(self.a)

    def oh(self):
        n = len(self.a)
        print()
        print('microblock_type = {:.2f} bits / entry'.format(self.microblock_type.nbytes * 8 / n))
        print('precomputed_queries = {:.2f} bits / entry'.format(self.precomputed_queries.nbytes * 8 / n))
        print('blocks = {:.2f} bits / entry'.format(self.blocks.nbytes * 8 / n))
        print('superblocks = {:.2f} bits / entry'.format(self.superblocks.nbytes * 8 / n))
        size = (self.microblock_type.nbytes +
                self.precomputed_queries.nbytes +
                self.blocks.nbytes +
                self.superblocks.nbytes) * 8
        return size / n


cdef class RMQ_stupid(RMQ_base):
    # attributes defined in rmq.pxd
    #cdef np.ndarray a

    def __init__(self, a):
        self.a = a

    cpdef uint32_t query(self, uint32_t i, uint32_t j) except -1:
        if i > j or j >= len(self.a):
            raise IndexError('invalid interval i={}, j={}'.format(i, j))

        cdef uint32_t pos, min_index, min_value
        cdef int32_t *a = <int32_t*>self.a.data

        min_index, min_value = i, a[i]

        for pos in range(i + 1, j + 1):
            if a[pos] < min_value:
                min_index = pos
                min_value = a[pos]

        return min_index

    def overhead(self):
        return 0.0


cpdef RMQ_base RMQ(np.ndarray[int32_t, ndim=1] lcp):
    if len(lcp) >= rmq_min_length:
        return RMQ_succinct(lcp)
    else:
        return RMQ_stupid(lcp)
