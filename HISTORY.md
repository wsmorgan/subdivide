# Revision History for "subdivide"

## Revision 0.1.7
- Added a new function that sorts the new areas to be in right to left
  order before they are returned by _create_new_areas. This solves the
  bug reported in [issue
  #12](https://github.com/wsmorgan/subdivide/issues/12)

## Revision 0.1.6
- Added another loop to _connect_segments that allows the code to
  search for a way to close an area using the newly created segments
  before moving on to extend the path.
-Added the ability to handle a more diverse number of possible
 trapezoids to the _trapezoid_cut routine.

## Revision 0.1.5
- Implemented changes to increase the number of subdivisions the code can go to.
- Refactored the _create_new_areas subroutine and created
  _connect_segments to make the code more readoble and to make
  debugging more efficient.
- Fixed vertex selection in _trapezoid_cut.

## Revision 0.1.4
- Fixed the vertex selection in _trapezoid_cut reported in issue #10.
- Started trying to resolve the dug described in issue #7.

## Revision 0.1.3
-Fixed the order of the returned points reported in issue #9.

## Revision 0.1.2
- Fixed subdivision of a triangle as reported in issue #3.
- Fixed the new point generation so thet if a projected point is
  essentially a vertex then the vertex is returned. Reported in issue #6.
- Added documentation to the README.md.

## Revision 0.1.1
- Converted the generators created by the zip functions to lists as described in issue #5.
- Added badges to the README.md.
- Fixed issues found by quantified code.

## Revision 0.1.0
- Added polygon.py to the repo.
- Initial commit.
