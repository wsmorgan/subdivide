# subdivide

Code for subdividing polygons and polyhedra into the desired number of
equal area or volume sections

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