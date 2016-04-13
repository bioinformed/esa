'''Experimental implementation of a suffix array data structure'''

from __future__ import division

from libc.string    cimport strcmp, strncmp
from cpython.string cimport PyString_FromStringAndSize

import numpy as np
cimport numpy as np

from rmq cimport RMQ, RMQ_base

from collections import defaultdict


cdef extern int sais_sa(char *T, int *SA, int n) except -1


cdef inline int min2(int a, int b):
    if a < b:
        return a
    else:
        return b


cdef inline int max2(int a, int b):
    if a > b:
        return a
    else:
        return b


cdef class ChildInterval(object):
    cdef readonly int lcp
    cdef readonly int lb
    cdef readonly int rb
    cdef readonly list children

    def __init__(self, int lcp=0, int lb=0, int rb=-1, list children=None):
        self.lcp = lcp
        self.lb = lb
        self.rb = rb
        self.children = children if children is not None else []

    def __repr__(self):
        return 'CI(lcp={}, lb={}, rb={}, children={})'.format(self.lcp, self.lb, self.rb, self.children)


cdef make_child_interval(int lcp=0, int lb=0, int rb=-1, list children=None):
    cdef ChildInterval c = ChildInterval.__new__(ChildInterval) # lcp, lb, rb, children)
    c.lcp = lcp
    c.lb = lb
    c.rb = rb
    c.children = children if children is not None else []
    return c


cdef class EnhancedSuffixArray(object):
    '''
    A suffix array is a sorted array of all suffixes of a string. It is a simple, yet powerful data
    structure which can be used to answer a wide variety of string-related queries.

    .. attribute:: text

    indexed text (plain str/bytes)

    .. attribute:: sa

    suffix array of text

    .. attribute:: isa

    reverse (inverse) of suffux array

    .. attribute:: lcp

    least common prefix array

    .. attribute:: rmq

    range mimimum query structure build on lcp array

    .. attribute:: bwt

    Burrows-Wheeler transform array of text
    '''
    cdef readonly bytes text, _bwt
    cdef readonly np.ndarray _sa, _isa, _lcp
    cdef RMQ_base _rmq

    def __init__(self, bytes text):
        self.text = text
        self._sa = self._isa = self._lcp = self._bwt = self._rmq = None

    property sa:
        '''
        suffix array of text

        sa satisfies the following invariant:

            all(text[sa[i-1]:] < text[sa[i]:] for i in range(1,len(sa)))
        '''
        def __get__(self):
            if self._sa is None:
                self._sa = self._compute_sa()
            return self._sa

    property isa:
        '''reverse (inverse) suffix array where sa[isa[i]] == isa[sa[i]] == i'''
        def __get__(self):
            if self._isa is None:
                self._isa = self._compute_isa()
            return self._isa

    property lcp:
        '''
        least common prefix array

        lcp[i]=n implies that the i'th string at sa[i] shares a common prefix of n characters with sa[i - 1].

        lcp satisfies the following invariants for i in range(1,len(sa)):

            text[sa[i-1]:sa[i-1]+lcp[i]] == text[sa[i]:sa[i]+lcp[i]]

            sa[i-1] + lcp[i] == len(text) or text[sa[i-1] + lcp[i]] < text[sa[i] + lcp[i]]

        '''
        def __get__(self):
            if self._lcp is None:
                self._lcp = self._compute_lcp()
            return self._lcp

    property rmq:
        '''range minimum query data structure built on lcp array'''
        def __get__(self):
            if self._rmq is None:
                self._rmq = RMQ(self.lcp)
            return self._rmq

    property bwt:
        '''Burrows-Wheeler transform array of text'''
        def __get__(self):
            if self._bwt is None:
                self._bwt = self._compute_bwt()
            return self._bwt

    def _compute_sa(self):
        '''
        Compute suffix array (sa) in O(n) time and at most 2n+O(1) working space using Yuta
        Mori's MIT licensed SAIS code.
        '''
        cdef int n = len(self.text)
        cdef char *text = <char *>self.text
        cdef np.ndarray sa_object = np.empty(n, dtype=np.intc)
        cdef int *sa = <int *>sa_object.data

        sais_sa(text, sa, n)

        return sa_object

    cdef _compute_isa(self):
        '''Compute reverse suffix array (isa) in O(n) time and O(n) space'''

        # NB: the following line has the side effect of materializing SA.  Do not move it!
        cdef np.ndarray isa_object = np.empty(len(self.sa), dtype=np.intc)
        cdef int *isa = <int *>isa_object.data
        cdef int *sa = <int *>self._sa.data
        cdef int i

        for i in range(len(self.sa)):
            isa[ sa[i] ] = i

        return isa_object

    cdef _compute_bwt(self):
        '''Compute Burrows-Wheeler transform array of text in O(n) time and O(n) space'''

        # NB: the following line has the side effect of materializing SA.  Do not move it!
        cdef bytes bwt_object = PyString_FromStringAndSize(<char *>NULL, len(self.sa))
        cdef char *bwt = <char *>bwt_object
        cdef char *text = <char *>self.text
        cdef int *sa = <int *>self._sa.data
        cdef int i, j

        for i in range(len(self.sa)):
            bwt[i] = text[sa[i] - 1] if sa[i] else '\0'

        return bwt_object

    cdef _compute_lcp(self):
        '''
        Compute least common prefix (lcp) array in O(n) time and O(n) space
        '''
        # NB: the following line has the side effect of materializing SA and RSA.  Do not move it!
        cdef int n      = len(self.isa)
        cdef np.ndarray lcp_object = np.empty(n + 1, dtype=np.intc)
        cdef int *lcp   = <int *>lcp_object.data
        cdef int *sa    = <int *>self._sa.data
        cdef int *isa   = <int *>self._isa.data
        cdef char *text = <char *>self.text
        cdef char *text1
        cdef char *text2
        cdef int i, k, h

        if n:
            lcp[0] = lcp[n] = 0

        h = 0
        for i in range(n):
            k = isa[i]

            if not k:
                h = 0
                continue

            text1 = text + i
            text2 = text + sa[k - 1]

            while text1[h] == text2[h]:
                h += 1

            lcp[k] = h

            if h:
                h -= 1

        return lcp_object

    cdef int sa_bisect_left(self, bytes query, int start=0, int stop=-1):
        self.sa

        cdef int *sa = <int *>self._sa.data
        cdef int m = len(query)
        cdef int mid

        if stop < 0:
            stop = len(self.text)

        cdef char *text = self.text
        cdef char *q    = query

        while start < stop:
            mid = start + ((stop - start) // 2) # avoid overflow
            if strncmp(q, text + sa[mid], m) > 0:
                start = mid + 1
            else:
                stop = mid

        return start

    cdef int sa_bisect_right(self, bytes query, int start=0, int stop=-1):
        self.sa

        cdef int *sa = <int *>self._sa.data
        cdef int m = len(query)
        cdef int mid

        if stop < 0:
            stop = len(self.text)

        cdef char *text = self.text
        cdef char *q    = query

        while start < stop:
            mid = start + ((stop - start) // 2) # avoid overflow
            if strncmp(q, text + sa[mid], m) < 0:
                stop = mid
            else:
                start = mid + 1

        return start

    def find_slow(self, bytes query):
        '''
        Generator yielding indices in text of query in O(m log n + occ) time
        where n is the length of text, m is the length of the query, and occ
        is the number of times query appears in text.
        '''
        cdef int start = 0
        cdef int stop  = len(self.text)

        # Ensure sa is materialized
        self.sa

        cdef int *sa = <int *>self._sa.data

        start = self.sa_bisect_left(query, start, stop)
        stop  = self.sa_bisect_right(query, start, stop)

        for i in range(start, stop):
            yield sa[i]

    cdef int get_children(self, char a, int *left, int *right) except -1:
        # Ensure sa, lcp and rmq are materialized
        self.rmq

        cdef int *lcp   = <int *>self._lcp.data
        cdef int *sa    = <int *>self._sa.data
        cdef char *text = <char *>self.text
        cdef int l, r, r_orig, ll

        l, r = left[0], right[0]
        left[0], right[0] = -1, -1

        r_orig, r = r, self._rmq.query(l + 1, r) if l < r else r
        ll = lcp[r]

        while l != r_orig and lcp[r] <= ll:
            if text[sa[l] + ll] == a:
                left[0], right[0] = l, r - 1
                return 0
            l, r = r, self._rmq.query(r + 1, r_orig) if r < r_orig else r_orig

        if text[sa[l] + ll] == a:
            left[0], right[0] = l, r_orig

        return 0

    def children(self, int left=0, right=-1):
        # Ensure sa, lcp and rmq are materialized
        self.rmq

        cdef int *lcp = <int *>self._lcp.data
        cdef int l, r, ll, n = len(self.text)

        if right == -1:
            right = len(self.text) - 1

        if left == right:
            return

        l, r = left, self._rmq.query(left + 1, right)
        ll = lcp[r]

        while l != right and lcp[r] <= ll:
            assert l < r
            yield lcp[self._rmq.query(l + 1, r - 1)] if l + 1 < r - 1 else n, l, r - 1
            l, r = r, self._rmq.query(r + 1, right)

        if l < right:
            yield lcp[self._rmq.query(l + 1, right)], l, right

    cdef void query_sa_interval(self, bytes query, int *left, int *right):
        # Ensure sa, lcp and rmq are materialized
        self.rmq

        cdef int *lcp   = <int *>self._lcp.data
        cdef int *sa    = <int *>self._sa.data
        cdef char *text = <char *>self.text
        cdef char *q    = <char *>query
        cdef int l, r, c, m, mm
        cdef bint found = True

        left[0] = right[0] = -1
        l = c = 0
        r = len(self.text) - 1
        m = len(query)

        while l != r and c != m and found:
            self.get_children(q[c], &l, &r)

            if l < 0:
                return

            mm = min2(lcp[self._rmq.query(l + 1, r)], m) if l != r else m
            found, c = strncmp(q + c, text + sa[l], mm - c) == 0, mm

        if found:
            left[0], right[0] = l, r + 1

    cpdef int occ(self, bytes query):
        '''
        Return the number of occurances of query in text in O(m) time where
        m is the length of the query.
        '''
        cdef int l, r
        self.query_sa_interval(query, &l, &r)
        return r - l

    def find(self, bytes query):
        '''
        Generator yielding indices in text of query in O(m + occ) time where
        m is the length of the query, and occ is the number of times query
        appears in text.
        '''
        cdef int *sa = <int *>self._sa.data
        cdef int l, r
        self.query_sa_interval(query, &l, &r)
        for i in range(l, r):
            yield sa[i]

    def postorder(self):
        self.lcp
        cdef int *lcp   = <int *>self._lcp.data
        cdef int *sa    = <int *>self._sa.data
        cdef int i, lb, ll, n
        cdef ChildInterval top, last_interval = None

        n = len(self.text)
        stack = [ChildInterval()]
        for i in range(n+1):
            lb = i - 1
            ll = lcp[i]
            top = stack[-1]

            while ll < top.lcp:
                last_interval = stack.pop()
                last_interval.rb = i - 1

                yield last_interval

                lb = last_interval.lb
                top = stack[-1]
                if ll <= top.lcp:
                    top.children.append(last_interval)
                    last_interval = None

            if ll > top.lcp:
                if last_interval is not None:
                    stack.append(make_child_interval(ll, lb, -1, [last_interval]))
                    last_interval = None
                else:
                    stack.append(make_child_interval(ll, lb, -1, []))
                top = stack[-1]

        while stack:
            last_interval = stack.pop()
            last_interval.rb = n - 1
            yield last_interval

    def preorder(self):
        self.rmq

        cdef int *lcp = <int *>self._lcp.data
        cdef int left = 0
        cdef int right = len(self.text) - 1
        cdef list stack = [make_child_interval(lcp[self._rmq.query(1, right - 1)], 0, right)]
        cdef ChildInterval top

        while stack:
            top = stack.pop()

            yield top

            if top.lb != top.rb:
                children = (make_child_interval(ll, l, r) for ll, l, r in self.children(top.lb, top.rb))
                stack += list(children)[::-1]


    def sfactors(self):
        '''
        Yield the Lempel-Ziv decomposition or s-factors of text in O(n) time and space.

        The Lempel-Ziv decomposition of s is the decomposition s = z1 z2 ... zl such that each zi
        is either a letter that does not occur in z1z2 ... zi−1 or the longest substring that
        occurs at least twice in z1z2 · · · zi (e.g., s = a·b·b·abbabb·c·ab·ab).  The substrings
        z1, z2, ..., zl are called the Lempel-Ziv factors or s-factors.  Factors are reported in
        the form of pairs (pi, |zi|), where pi is either the position of a nontrivial occurrence of
        zi in z1z2 ... zi (it is called an earlier occurrence of zi) or zi itself if zi is a
        letter that does not occur in z1z2 ... zi−1.  The reported pairs are generated
        incrementally are not stored in main memory.

        >>> list(EnhancedSuffixArray('abaababa').sfactors())
        [((0, 1), 'a'), (1, 1, 'b'), (0, 1, 'a'), (0, 3, 'aba'), (1, 2, 'ba')]
        >>> list(EnhancedSuffixArray('1100101010000').sfactors())
        [((0, 1), '1'), (0, 1, '1'), (2, 1, '0'), (2, 1, '0'), (1, 2, '10'), (4, 4, '1010'), (9, 3, '000')]
        >>> list(EnhancedSuffixArray('GTTGTTGTTGTT').sfactors())
        [((0, 1), 'G'), (1, 1, 'T'), (1, 1, 'T'), (0, 9, 'GTTGTTGTT')]
        >>> list(EnhancedSuffixArray('abbabbabbcabab').sfactors())
        [((0, 1), 'a'), (1, 1, 'b'), (1, 1, 'b'), (0, 6, 'abbabb'), (9, 1, 'c'), (0, 2, 'ab'), (0, 2, 'ab')]

        The algorithm is adapted from CPS2 algorithm of Chen et al. (2008) modified to use
        O(|sigma|) interval refinement using RMQ data structure on the LCP array, resulting in a
        O(n |sigma|) = O(n) algorithm.  Several minor errors from the pseudocode outlined in the
        paper were corrected.

        Developer note: During processing, the RMQ data structure for SA is computed, RMA_SA.  Note
        that RMQ_SA is distinct from RMQ_LCP, which is used for finding child and parent links in
        this implementation of Enhanced Suffix Arrays.  RMA_SA is computed locally to this function
        and not stored.  If RMQ_SA is needed elsewhere, then the implementation should be altered
        to cached it along with other expensive computed data structures.

        See:
            Chen, Gang, Simon J.  Puglisi, and William F.  Smyth.
            "Lempel–Ziv factorization using less time & space." Mathematics
            in Computer Science 1.4 (2008): 605-623.

            Chen, Gang, Simon J.  Puglisi, and William F.  Smyth.  "Fast and
            practical algorithms for computing all the runs in a string."
            Combinatorial Pattern Matching.  Springer Berlin Heidelberg,
            2007.

            Main, M.G.: Detecting leftmost maximal periodicities.  Discrete
            Applied Maths 25, 145–153 (1989)
        '''
        self.rmq
        rmq_sa = RMQ(self.sa)

        cdef char *text = <char *>self.text
        cdef int  *sa   = <int *>self._sa.data
        cdef int  *lcp  = <int *>self._lcp.data
        cdef int i, j, min_match, left, right, lzpos, lzlen, n = len(self.text)

        yield (0, 1), self.text[0]

        i = 1
        while i < n:
            lzpos = j = i
            left, right, min_match = 0, n - 1, -1

            while min_match < i and j < n:
                self.get_children(text[j], &left, &right)
                ll = lcp[self.rmq.query(left + 1, right)] if left != right else (n - i)
                min_match = sa[rmq_sa.query(left, right)]
                if min_match < i:
                    lzpos = min_match
                    j = i + ll

            lzlen = max2(1, j - i)
            yield lzpos, lzlen, text[lzpos:lzpos+lzlen]
            i += lzlen


    def __contains__(self, bytes query):
        '''Return True if text contains query in O(m) time where is is the length of the query'''
        cdef int l, r
        self.query_sa_interval(query, &l, &r)
        return l != r

    def check_integrity_slow_old_dont_use(self):
        '''Check integrity of sa, isa and lcp'''

        self.lcp

        cdef char *text = <char *>self.text
        cdef int *sa    = <int *>self._sa.data
        cdef int *isa   = <int *>self._isa.data
        cdef int *lcp   = <int *>self._lcp.data
        cdef int i, n = len(self.text)

        for i in range(1, n):
            if strcmp(text + sa[i-1], text + sa[i]) >= 0:
                raise ValueError('Suffix array ordering error')

        for i in range(n):
            if isa[sa[i]] != i or sa[isa[i]] != i:
                raise ValueError('Suffix array inverse error')

        for i in range(1, n):
            if strncmp(text + sa[i-1], text + sa[i], lcp[i]) != 0:
                raise ValueError('Least common prefix mismatch error')

        for i in range(1, n):
            if sa[i-1] + lcp[i] < n and text[sa[i-1] + lcp[i]] >= text[sa[i] + lcp[i]]:
                raise ValueError('Suffix array/least common prefix error')


    def check_integrity(self):
        '''Check integrity of sa, isa and lcp'''

        self.lcp

        cdef char *text = <char *>self.text
        cdef int *sa    = <int *>self._sa.data
        cdef int *isa   = <int *>self._isa.data
        cdef int *lcp   = <int *>self._lcp.data
        cdef int i, j, k, n = len(self.text)

        for i in range(n):
            # Condition 1
            if sa[i] < 0 or sa[i] >= n:
                raise ValueError('Suffix array value out of bounds')

            if isa[i] < 0 or isa[i] >= n:
                raise ValueError('Inverse suffix array value out of bounds')

            # Condition 2: Integrity of isa
            if isa[sa[i]] != i or sa[isa[i]] != i:
                raise ValueError('Suffix array inverse error')

            if i == 0:
                continue

            # Condition 3: Ordering of first character of each suffix
            if text[sa[i - 1]] > text[sa[i]]:
                raise ValueError('Suffix array values out of order')

            # Condition 4:
            if text[sa[i - 1]] == text[sa[i]] and sa[i - 1] != n - 1:
                j = isa[sa[i - 1] + 1]
                k = isa[sa[i] + 1]
                if sa[j] != sa[i - 1] + 1 or sa[k] != sa[i] + 1 or j >= k:
                    raise ValueError('Suffix array invalid')

            if sa[i-1] + lcp[i] < n and text[sa[i-1] + lcp[i]] >= text[sa[i] + lcp[i]]:
                raise ValueError('Suffix array/least common prefix error')


def longest_repeated_substring(bytes text):
    '''
    Return all instances of the longest repeated substring within text and their positions.

    >>> sorted(longest_repeated_substring('banana'))
    [(1, 'ana'), (3, 'ana')]

    >>> text = "not so Agamemnon, who spoke fiercely to "
    >>> sorted(longest_repeated_substring(text))
    [(0, 'no'), (3, ' s'), (5, 'o '), (13, 'no'), (20, 'o '), (21, ' s'), (38, 'o ')]

    This function can be easy modified for any criteria, e.g. for searching ten
    longest non overlapping repeated substrings.
    '''
    index = EnhancedSuffixArray(text)
    cdef int i
    cdef bint first

    sa  = index.sa
    lcp = index.lcp

    maxlen = index.lcp.max()

    # No repeated strings present, so bail
    if not maxlen:
        return

    first = True
    for i in range(1, len(text)):
        if lcp[i] == maxlen:
            substring = text[sa[i]:sa[i] + lcp[i]]
            if first:
                first = False
                yield sa[i - 1], substring
            yield sa[i], substring
        else:
            first = True


cpdef longest_common_substring(bytes text1, bytes text2):
    '''
    Returns the longest common substring in O(len(text1) + len(text2)) time and space.

    >>> longest_common_substring('ABABC', 'BABCA')
    'BABC'

    >>> longest_common_substring('AAAAA', 'AAAAAAA')
    'AAAAA'

    >>> longest_common_substring('Neque porro quisquam est qui dolorem ipsum quia dolor sit amet',
    ...                          'Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris')
    ' quis'
    '''
    text = text1 + '\0' + text2

    cdef EnhancedSuffixArray index = EnhancedSuffixArray(text)

    # materialize LCP array
    index.lcp

    cdef int *sa = <int *>index._sa.data
    cdef int *lcp = <int *>index._lcp.data

    cdef int len1 = len(text1)
    cdef int best_len = 0
    cdef int best_pos = 0
    cdef int i

    # Skip index 0, since that contains the delimiter symbol
    # Uses a look-behind of 1, so iteration starts at 2
    for i in range(2,len(text)):
        if (sa[i-1] < len1) != (sa[i] < len1) and lcp[i] > best_len:
            best_len = lcp[i]
            best_pos = sa[i]

    return text[best_pos:best_pos + best_len]


def longest_palendrome(bytes text):
    '''
    Return the longest palendrome in text

    >>> longest_palendrome('mississippi')
    'ississi'
    >>> longest_palendrome('drab as a fool aloof as a bard'.replace(' ',''))
    'drabasafoolaloofasabard'
    '''
    return longest_common_substring(text, text[::-1])


def minimum_lexical_rotation(bytes text):
    if not text:
        return ''
    elif len(text) == 1:
        return text

    double_text = text + text

    cdef EnhancedSuffixArray index = EnhancedSuffixArray(double_text)

    # materialize SA
    index.sa

    cdef int *sa = <int *>index._sa.data
    cdef int n = len(text)
    cdef int i, j

    for i in range(n):
        j = sa[i]
        if j < n:
            return double_text[j:j+n]

    raise ValueError('impossible')


def supermaximal_repeats(bytes text):
    if not text:
        return ''
    elif len(text) == 1:
        return text

    cdef EnhancedSuffixArray index = EnhancedSuffixArray(text)

    # materialize LCP array
    index.lcp

    cdef char *s = <char *>text
    cdef int *sa = <int *>index._sa.data
    cdef int *lcp = <int *>index._lcp.data
    cdef int n = len(text)
    cdef int i, j, ll, l, lr

    i = 0
    while i < n:
        j  = i + 1
        ll = lcp[i]
        l  = lcp[j] if j < n else 0

        while j < n-1 and lcp[j+1] == l:
            j += 1

        lr = lcp[j + 1] if j + 1 < n else 0

        i = max2(i + 1, j - 1)


def maximum_unique_matches(bytes text1, bytes text2):
    cdef int p = len(text1)
    text = text1 + '\0' + text2

    cdef EnhancedSuffixArray index = EnhancedSuffixArray(text)

    # materialize LCP array
    index.lcp

    cdef char *s = <char *>text
    cdef int *sa = <int *>index._sa.data
    cdef int *lcp = <int *>index._lcp.data
    cdef int n = len(text)
    cdef int i, j, ll, l, lr
    cdef char left1, left2

    i = 0
    while i < n:
        j  = i + 1
        ll = lcp[i]
        l  = lcp[j] if j < n else 0

        while j < n-1 and lcp[j+1] == l:
            j += 1

        lr = lcp[j + 1] if j + 1 < n else 0

        if l and lr < l and j == i+1:
            left1 = s[sa[i]-1] if sa[i] else '\1'
            left2 = s[sa[j]-1] if sa[j] else '\1'
            if (sa[i] < p) != (sa[j] < p) and left1 != left2:
                yield text[sa[i]:sa[i]+l]

        i = max2(i + 1, j - 1)


def tandem_repeats(bytes text):
    cdef EnhancedSuffixArray index = EnhancedSuffixArray(text)

    # materialize LCP array
    index.lcp

    cdef char *s  = <char *>text
    cdef int *sa  = <int *>index._sa.data
    cdef int *isa = <int *>index._isa.data
    cdef int i, j, l, q, p, n = len(text)
    cdef int left

    for ci in index.postorder():
        i, j, l = ci.lb, ci.rb, ci.lcp
        if not l:
            continue
        for q in range(i, j+1):
            p = sa[q]
            if p+2*l <= n and i <= isa[p+l] <= j and s[p+l] != s[p+2*l]:
                left = p
                while left and s[left - 1] == s[left + l - 1]:
                    left -= 1
                right = p + 2*l
                #yield (p, l), s[p:p+l]+':'+s[p+l:p+2*l]
                yield (left, right, l, s[p:p+l], (right - left) // l, (right - left) % l)


def tandem_repeats2(bytes text):
    cdef EnhancedSuffixArray index = EnhancedSuffixArray(text)

    # materialize LCP array
    index.lcp

    cdef char *s  = <char *>text
    cdef int *sa  = <int *>index._sa.data
    cdef int *isa = <int *>index._isa.data
    cdef int i, j, l, ll, q, p, n = len(text)
    cdef int left, right

    found = defaultdict(list)

    cdef int m = 0
    for ci in index.preorder():
        i, j, l = ci.lb, ci.rb, ci.lcp
        if not l:
            continue
        for q in range(i, j+1):
            m += 1
            p = sa[q]
            if p+2*l <= n and i <= isa[p+l] <= j and s[p+l] != s[p+2*l]:
                right = p + 2*l
                skip = False
                if right in found:
                    for llen, ll in found[right]:
                        if llen >= l and l % ll == 0:
                            skip = True
                            continue

                if skip:
                    continue

                left = p
                # Trace backward by checking each previous l characters are also within the child interval
                while l > 1 and left >= l and i <= isa[left - l] <= j:
                    left -= l

                while left and s[left - 1] == s[left + l - 1]:
                    left -= 1

                found[right].append( (right - left, l) )
                #yield (p, l), s[p:p+l]+':'+s[p+l:p+2*l]
                yield (left, right, l, (right - left) // l, (right - left) % l, s[left:left+l])
