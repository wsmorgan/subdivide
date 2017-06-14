[![Build Status](https://travis-ci.org/wsmorgan/subdivide.svg?branch=master)](https://travis-ci.org/wsmorgan/subdivide)[![codecov](https://codecov.io/gh/wsmorgan/subdivide/branch/master/graph/badge.svg)](https://codecov.io/gh/wsmorgan/subdivide)[![Code Issues](https://www.quantifiedcode.com/api/v1/project/7f68b7021a174e4da28e14e8b4b31379/badge.svg)](https://www.quantifiedcode.com/app/project/7f68b7021a174e4da28e14e8b4b31379)

# subdivide

Code for subdividing polygons and polyhedra into the desired number of
equal area or volume sections

Full API Documentation available at: [github pages](https://wsmorgan.github.io/subdivide/).


## Installing the code

To intall the code clone this repository and execute the following
command in the repositories root directory:

```
python setup.py install
```

## Running the code

At present only the polygon subdivision option is available. To use it
the user must define the vertices of the polygon in counterclockwise
oredr. For example to subdivide a square:

```
from subdivide.polygon import Polygon
verts = [[0,0],[0,1],[1,1],[1,0]]
p = Polygon(verts)
p.subdivide(4)
```