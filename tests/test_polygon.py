"""Tests the polygon module.
"""
import pytest
import os
import numpy as np
from subdivide.polygon import Polygon

def test_polygon():
    """Tests the initialization of the Polygon class.
    """
    verts = [[0,0],[1,0],[1,1],[0,1]]
    p = Polygon(verts)
    assert 1.0 == p.area
    assert verts == p.vertices
    assert [([0, 0], [1, 0]), ([1, 0], [1, 1]), ([1, 1], [0, 1]), ([0, 1], [0, 0])] == p._segments
    assert 4.0 == p.perimiter

    verts = [[0,0],[1,0],[1,1]]
    segments = p._find_segments(verts=verts)
    assert [([0, 0], [1, 0]), ([1, 0], [1, 1]), ([1, 1], [0, 0])] == segments
    assert 0.5 == p._find_area(segments=segments)
    assert np.allclose(3.41421356237,p._find_permimter(segments=segments))

def test_projection():
    """Tests the projection function of the Polygon class.
    """

    verts = [[0,0],[1,0],[1,1]]
    p = Polygon(verts)
    line1 = ([0,0],[1,0])
    line2 = ([1,0],[1,1])
    bisect = np.array([[1,0],[0.5,0.5]])
    proj = [[[1,1],[0,0]],[[0,0],[1,1]]]

    assert p._projections(line1,line2,bisect) == proj
    
def test_subdived():
    """Tests the subdivide function of the Polygon class.
    """

    verts = [[0,0],[1,0],[1,1],[0,1]]
    p = Polygon(verts)
    out2 = [[[0.0, 0.5], [0, 0], [1, 0], [1.0, 0.5]], [[0, 1], [0.0, 0.5], [1.0, 0.5], [1, 1]]]
    print("test out",p.subdivide(2))
    assert out2 == p.subdivide(2)

    verts = [[0,0],[1,0],[1,1]]
    p = Polygon(verts)

    out2 = [[[0.5, 0.5], [1, 0], [1, 1]], [[0, 0], [1, 0], [0.5, 0.5]]]
    print("test out",p.subdivide(2))
    assert out2 == p.subdivide(2)

    with pytest.raises(TypeError):
        p.subdivide(2.1)

def test_finde_centroid():
    """Tests the find centroid function of the Polygon class.
    """

    points = [[0,0],[0,1],[1,1],[1,0]]
    p = Polygon(points)
    assert [0.5,0.5] == p._find_centroid(points)

    points = [[0,0],[0,1],[1,1]]
    assert np.allclose([0.3333333333333333,0.6666666666666667],p._find_centroid(points))

    points = [[0,0],[1,0],[0.5,1]]
    assert np.allclose([0.5,0.3333333333333333],p._find_centroid(points))

def test_cc_sort():
    """Tests the sorting of points to be in counterclockwise oredr in the Polygon class.
    """

    verts = [[0,0],[1,0],[1,1],[0,1]]
    p = Polygon(verts)

    points = [[0,0],[0,1],[1,1],[1,0]]

    assert [[0,1],[0,0],[1,0],[1,1]] == p._counter_clockwise_sort(points)

    points = [[1,1],[0,0],[0.5,2],[1,0],[0,1]]
    assert [[0, 1], [0, 0], [1, 0], [1, 1], [0.5, 2]] == p._counter_clockwise_sort(points)
