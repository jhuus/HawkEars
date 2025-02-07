# Unit tests for base_tester.py.
# TODO: add more tests.

from types import SimpleNamespace

import pytest

from base_tester import BaseTester

def test_get_offsets_no_overlap():
    tests = [
        SimpleNamespace(start=0, end=9, expected=[0, 3, 6]),
        SimpleNamespace(start=1, end=9, expected=[0, 3, 6]),
        SimpleNamespace(start=2.6, end=9, expected=[0, 3, 6]),
        SimpleNamespace(start=2.8, end=9, expected=[3, 6]),
        SimpleNamespace(start=3.7, end=8.8, expected=[3, 6]),
        SimpleNamespace(start=3.7, end=6.31, expected=[3, 6]),
        SimpleNamespace(start=3.7, end=6.2, expected=[3]),
        SimpleNamespace(start=0, end=0, expected=[]),
    ]
    bt = BaseTester()

    for test in tests:
        offsets = bt.get_offsets(test.start, test.end)
        assert(len(offsets) == len(test.expected))
        assert(offsets[i] == test.expected[i] for i in range(len(offsets)))

def test_get_offsets_overlap():
    tests = [
        SimpleNamespace(start=0, end=9, overlap=1.5, expected=[0, 1.5, 3, 4.5, 6, 7.5]),
        SimpleNamespace(start=1, end=9, overlap=1.5, expected=[0, 1.5, 3, 4.5, 6, 7.5]),
        SimpleNamespace(start=2.6, end=9, overlap=1.5, expected=[0, 1.5, 3, 4.5, 6, 7.5]),
        SimpleNamespace(start=2.8, end=9, overlap=1.5, expected=[1.5, 3, 4.5, 6, 7.5]),
        SimpleNamespace(start=4.3, end=9, overlap=1.5, expected=[3, 4.5, 6, 7.5]),
        SimpleNamespace(start=4.1, end=6.31, overlap=1.5, expected=[1.5, 3.0, 4.5, 6]),
        SimpleNamespace(start=4.1, end=6.2, overlap=1.5, expected=[1.5, 3, 4.5]),
    ]
    bt = BaseTester()

    for test in tests:
        offsets = bt.get_offsets(test.start, test.end, overlap=test.overlap)
        assert(len(offsets) == len(test.expected))
        assert(offsets[i] == test.expected[i] for i in range(len(offsets)))

def test_get_segments_no_overlap():
    tests = [
        SimpleNamespace(start=0, end=9, expected=[0, 1, 2]),
        SimpleNamespace(start=1, end=9, expected=[0, 1, 2]),
        SimpleNamespace(start=2.6, end=9, expected=[0, 1, 2]),
        SimpleNamespace(start=2.8, end=9, expected=[1, 2]),
        SimpleNamespace(start=3.7, end=8.8, expected=[1, 2]),
        SimpleNamespace(start=3.7, end=6.31, expected=[1, 2]),
        SimpleNamespace(start=3.7, end=6.2, expected=[1]),
    ]
    bt = BaseTester()

    for test in tests:
        offsets = bt.get_segments(test.start, test.end)
        assert(len(offsets) == len(test.expected))
        assert(offsets[i] == test.expected[i] for i in range(len(offsets)))

def test_get_segments_overlap():
    tests = [
        SimpleNamespace(start=0, end=9, overlap=1.5, expected=[0, 1, 2, 3, 4, 5]),
        SimpleNamespace(start=1, end=9, overlap=1.5, expected=[0, 1, 2, 3, 4, 5]),
        SimpleNamespace(start=2.6, end=9, overlap=1.5, expected=[0, 1, 2, 3, 4, 5]),
        SimpleNamespace(start=2.8, end=9, overlap=1.5, expected=[1, 2, 3, 4, 5]),
        SimpleNamespace(start=4.3, end=9, overlap=1.5, expected=[2, 3, 4, 5]),
        SimpleNamespace(start=4.1, end=6.31, overlap=1.5, expected=[1, 2, 3, 4]),
        SimpleNamespace(start=4.1, end=6.2, overlap=1.5, expected=[1, 2, 3]),
        SimpleNamespace(start=0, end=1.5, overlap=1.5, expected=[0]),
    ]
    bt = BaseTester()

    for test in tests:
        offsets = bt.get_segments(test.start, test.end, overlap=test.overlap)
        assert(len(offsets) == len(test.expected))
        assert(offsets[i] == test.expected[i] for i in range(len(offsets)))
