"""This module contains methods needed to subdivide a polygon into N subpolyons of equal area."""

import numpy as np
from matplotlib import path 

class Polygon(object):
    """A representation for a polygon and it's devisions.

    Attributes:
        area (float): The area of the polygon.
        vertices (list of float): The vertices of the polygon. 
        perimiter (float): The perimiter of the polygon. 
        sub_polgons (list of list): The list of the vertices of the subpolygons
    """

    def __init__(self,vertices,eps_=None):
        """Initializes the polygon.

        Args:
            vertices (list of float): The verticies of the polgon listed in 
              counterclockwise order.
           eps (optional float): Floating point accuracy, default 1E-7.
        """

        self.vertices = vertices
        self._segments = self._find_segments()
        self.area = self._find_area()
        self.perimiter = self._find_permimter()
        self._path = path.Path(vertices)
        if eps_ is None:
            self._eps = 1E-7
        else:
            self._eps = eps_


    def subdivide(self,N):
        """Subdivides the polygon into segments of equal area with minimal perimiters.

        Args:
            N (int): The number of subdivisions desired.
        
        Returns:
            sub_polys (list): A list of the vertices of the subpolygons found.

        Raises:
            TypeError: If N is not an integer.
        """
        from copy import deepcopy
        sub_polys = []
        if isinstance(N,int):
            subarea = self.area/N
            remaining_verts = deepcopy(self.vertices)
            while len(sub_polys) < N-1:
                sub_poly, remaining_verts = self._find_subdivision(remaining_verts,subarea)
                sub_polys.append(sub_poly)
            sub_polys.append(remaining_verts)
        else:
            raise TypeError("Please specify an integer number of subdivisions of the "
                             "polygon.")

        return sub_polys

    def _find_subdivision(self,verts,area):
        """Finds a subdivision of the desired area for the polygon formed by the vertices

        Args:
            verts (list): The vertices of the target polygon.
            area (float): The area for the new sub polygon.
        
        Returns:
            sub_poly (list): The vertices of the polygon with the desired area.
            remaining_poly (list): The vertices definining the remaining polygon.
        """
        from copy import deepcopy

        segments = self._find_segments(verts=verts)
        total_area = self._find_area(segments=segments)
        n_segs = len(segments)
        sub_poly_per = None
        
        for i in range(n_segs):
            for j in range(n_segs):
                if i != j:
                    new_segments, bisector, self._new_points = self._create_new_segments(segments[i],segments[j])
                    divisions = self._create_new_areas(new_segments,segments)
                    test_poly = None

                    print("seg_i",segments[i])
                    print("seg_j",segments[j])
                    print("new segs",new_segments)
                    print("divisions",divisions)
                    if len(divisions) == 1 and self._find_area(divisions[0]) > area:
                        if len(divisions[0]) == 3:
                            test_poly = self._triangle_cut(divisions[0],area,None)
                        elif len(divisions[0]) == 4:
                            test_poly = self._trapezoid_cut(divisions[0],area,bisector,None)
                    elif len(divisions) == 2:
                        if self._find_area(divisions[0]) > area:
                            if len(divisions[0]) == 3:
                                test_poly = self._triangle_cut(divisions[0],area,None)
                            elif len(divisions[0]) == 4:
                                test_poly = self._trapezoid_cut(divisions[0],area,bisector,None)
                        if self._find_area(divisions[1]) > area and test_poly is None:
                            if len(divisions[1]) == 3:
                                test_poly = self._triangle_cut(divisions[1],area,None)
                            elif len(divisions[1]) == 4:
                                test_poly = self._trapezoid_cut(divisions[1],area,bisector,None)
                    elif len(divisions) == 3:
                        area1 = self._find_area(segments=divisions[0])
                        area2 = self._find_area(segments=divisions[1])
                        area3 = self._find_area(segments=divisions[2])
                        if area1 > area:
                            if len(divisions[0]) == 3:
                                test_poly = self._triangle_cut(divisions[0],area,None)
                            elif len(divisions[0]) == 4:
                                test_poly = self._trapezoid_cut(divisions[0],area,bisector,None)
                        elif (area1 + area2) > area:
                            if len(divisions[1]) == 3:
                                test_poly = self._triangle_cut(divisions[1],area,divisions[0])
                            elif len(divisions[1]) == 4:
                                test_poly = self._trapezoid_cut(divisions[1],area,bisector,divisions[0])
                        if area3 > area and test_poly is None:
                            if len(divisions[2]) == 3:
                                test_poly = self._triangle_cut(divisions[2],area,None)
                            elif len(divisions[2]) == 4:
                                test_poly = self._trapezoid_cut(divisions[2],area.bisector,None)
                    else:
                        raise RuntimeError("The code found more than 3 divisions of "
                                           "the polygon for a given pair of line "
                                           "segments. This should not be possible.")
                    if test_poly is not None:
                        test_segs = self._find_segments(verts=test_poly)
                        if sub_poly_per is None:
                            if abs(self._find_area(segments=test_segs) - area) < self._eps:
                                sub_poly = test_poly
                                sub_poly_per = self._find_permimter(segments=test_segs)
                        else:
                            if abs(self._find_area(segments=test_segs) - area) < self._eps and self._find_permimter(segments=test_segs) < sub_poly_per:
                                sub_poly = test_poly
                                sub_poly_per = self._find_permimter(segments=test_segs)

        # Add the vertices for the newly found polygon to the list so
        # that we can determine what the remaining larger polygon looks like.
        temp_verts = deepcopy(verts)
        for vert in sub_poly:
            if vert not in temp_verts:
                loc = 0
                new_loc = None
                while new_loc is None:
                    if loc == len(temp_verts)-1:
                        next_loc = 0
                    else:
                        next_loc = loc+1

                    if self._is_between(temp_verts[loc],temp_verts[next_loc],vert):
                        new_loc = next_loc
                    else:
                        loc += 1

                t_size = 0
                orig_list = 0
                temp = []
                while len(temp) < len(temp_verts)+1:
                    if new_loc == t_size:
                        temp.append(vert)
                        t_size += 1
                    else:
                        temp.append(temp_verts[orig_list])
                        t_size += 1
                        orig_list += 1
                temp_verts = temp 

        # Now we need to figure out which vertices form the remaining volume.
        from itertools import combinations

        remaining_poly = []
        for vert in temp_verts:
            if vert in sub_poly and vert not in verts:
                remaining_poly.append(vert)
            elif vert in verts and vert not in sub_poly:
                remaining_poly.append(vert)                    

        return sub_poly, remaining_poly

    def _create_new_segments(self,line1,line2):
        """Finds the projection of the endpoints of the lines across the angle
        bisector of the lines. Then uses those points to construct the
        new possible segments of the input lines.

        Args:
            line1 (list): The endpoints of the first line.
            line2 (list): The endpoints of the second line.

        Returns:
            new_segments (list): The new segments formed by the new points.
            bisector (list): The starting and ending points of the bisector.
            new_verts (list): A list of the new vertices.
        """

        bisector = self._angle_bisection(line1,line2)
        new_points = self._projections(line1,line2,bisector)

        print("new_points",new_points)
        new_segments = []

        if len(new_points) == 2 and ((np.allclose(new_points[0][0],new_points[1][1]) and np.allclose(new_points[0][1],new_points[1][0])) or (np.allclose(new_points[0][0],new_points[1][0]) and np.allclose(new_points[0][1],new_points[1][1]))):
            new_points = [new_points[0]]

        for point in new_points:
            if not (np.allclose(point[0],line1[0]) or np.allclose(point[0],line1[1])) and not (np.allclose(point[0],line2[0]) or np.allclose(point[0],line2[1])):
                if self._is_between(line1[0],line1[1],point[0]):
                    new_segments.append((line1[0],point[0]))
                    new_segments.append((point[0],line1[1]))
                elif self._is_between(line2[0],line2[1],point[0]):
                    new_segments.append((line2[0],point[0]))
                    new_segments.append((point[0],line2[1]))
            else:
                if (np.allclose(point[1], line1[0]) or np.allclose(point[1],line1[1])) and not (np.allclose(point[0],line1[0]) or np.allclose(point[0],line1[1])):
                    new_segments.append((point[1],point[0]))
                elif (np.allclose(point[1], line2[0]) or np.allclose(point[1],line2[1])) in line2 and not (np.allclose(point[0],line2[0]) or np.allclose(point[0],line2[1])):
                    new_segments.append((point[0],point[1]))

        for point in new_points:
            if (point[0],point[1]) not in new_segments or (point[1],point[0]) not in new_segments:
                option1 = (point[1][0]-point[0][0])*(point[1][1]+point[0][1])
                if option1 < 0:
                    new_segments.append((point[0],point[1]))
                else:
                    new_segments.append((point[1],point[0]))
                    
        return new_segments, bisector, new_points
    
    @staticmethod
    def _create_new_areas(new_segments,old_segments):
        """Uses the newly created segments to diffine subvolumes of the polygon.

        Args:
            new_segments (list): The list of segments created by 
              self._create_new_segments.
            old_segments (list): A list containing the other segments 
              needed to form the polygon.

        Returns:
            new_areas (list): A list of lists of segments where each list defines a 
              new polgon of smaller area.
        """

        all_segments = new_segments+old_segments

        new_segments_local = []
        for seg in new_segments:
            new_segments_local.append(seg)
            new_segments_local.append(seg[::-1])

        new_areas = []

        for seg in new_segments_local:
            new_path =[seg]
            cur_seg = seg
            for test_seg in all_segments:
                if not (np.allclose(test_seg[0],cur_seg[0]) and np.allclose(test_seg[1],cur_seg[1])) and not (np.allclose(test_seg[0],cur_seg[1]) and np.allclose(test_seg[1],cur_seg[0])) and np.allclose(test_seg[0],cur_seg[1]):
                    if not test_seg[0]==cur_seg[1]:
                        cur_seg = list(cur_seg)
                        cur_seg = cur_seg[:-1]
                        cur_seg.append(test_seg[0])
                        list(cur_seg)[1] = test_seg[0]
                        new_path = new_path[:-1]
                        new_path.append(tuple(cur_seg))

                    new_path.append(test_seg)
                    cur_seg = test_seg
                    if np.allclose(cur_seg[1],seg[0]):
                        break
                        
            if not np.allclose(cur_seg[1],seg[0]):
                for test_seg in all_segments:
                    if not (np.allclose(test_seg[0],cur_seg[0]) and np.allclose(test_seg[1],cur_seg[1])) and not (np.allclose(test_seg[0],cur_seg[1]) and np.allclose(test_seg[1],cur_seg[0])) and np.allclose(test_seg[0],cur_seg[1]):
                        if not test_seg[0]==cur_seg[1]:
                            cur_seg = list(cur_seg)
                            cur_seg = cur_seg[:-1]
                            cur_seg.append(test_seg[0])
                            list(cur_seg)[1] = test_seg[0]
                            new_path = new_path[:-1]
                            new_path.append(tuple(cur_seg))

                        new_path.append(test_seg)
                        cur_seg = test_seg
                        if np.allclose(cur_seg[1],seg[0]):
                            break
                
            if not new_path in new_areas and len(new_path) > 2:
                new_areas.append(new_path)

        return new_areas

    def _triangle_cut(self,segments,total_area,rest_of_poly):
        """Finds the desired cut inside a triangle to get the correct area.
        
        Args:
            segments (list): The line segments that form the triangel.
            total_area (float): The area desired after the cut.
            rest_of_poly (list): The line segments containing the rest of the
              polygon whose area will contribute.

        Returns:
            poly (list): The vertices that form the polygon with the desired area.

        Raises:
            RunTimeError: A RunTimeError is raised if the corroct area cannot be found.
        """

        if rest_of_poly is None:
            target = total_area
        else:
            target = total_area - self._find_area(rest_of_poly)

        at = segments[0][0]
        bt = segments[1][0]
        ct = segments[2][0]

        a = None
        for point in self._new_points:
            if at == point[1]:
                a = at
                b = bt
                c = ct
            elif bt == point[1]:
                a = bt
                b = ct
                c = at
            elif ct == point[1]:
                a = ct
                b = at
                c = bt

            if a is not None:
                break

        cut_point = list(np.array(c) + target/self._find_area(segments=segments)*(np.array(b)-np.array(c)))

        if rest_of_poly is None:
            poly = [a,cut_point,c]
        else:
            poly = []
            for seg in rest_of_poly:
                if seg[0] == a:
                    poly.append(a)
                    poly.append(cut_point)
                    poly.append(c)
                elif seg[0] == c:
                    poly.append(c)
                    poly.append(b)
                    poly.append(a)
                else:
                    poly.append(seg[0])
                    
        if abs(self._find_area(segments=self._find_segments(verts=poly))-total_area) > self._eps:
            raise RuntimeError("Failed to find a cut line for the target area in "
                               "triange_cut.")

        return poly

    def _trapezoid_cut(self,segments,total_area,bisector,rest_of_poly):
        """Finds the desired cut inside a trapezoid to get the correct area.
        
        Args:
            segments (list): The line segments that form the trapezoid.
            total_area (float): The area desired after the cut.
            bisector (list): The endpoints of the bisector.
            rest_of_poly (list): The line segments containing the rest of the
              polygon whose area will contribute.

        Returns:
            poly (list): The vertices that form the polygon with the desired area.

        Raises:
            RunTimeError: A RunTimeError is raised if the corroct area cannot be found.
        """
        if rest_of_poly is None:
            target = total_area
        else:
            target = total_area - self._find_area(rest_of_poly)

        at = segments[0][0]
        bt = segments[1][0]
        ct = segments[2][0]
        dt = segments[3][0]

        a = None
        for point in self._new_points:
            if at == point[1]:
                a = at
                b = bt
                c = ct
                d = dt
            elif bt == point[1]:
                a = bt
                b = ct
                c = dt
                d = at
            elif ct == point[1]:
                a = ct
                b = dt
                c = at
                d = bt
            elif dt == point[1]:
                a = dt
                b = at
                c = bt
                d = ct

            if a is not None:
                break

        ad = np.array(d)-np.array(a)
        bc = np.array(c)-np.array(b)
        bi_v = self._unit_vec(np.array(bisector[1]),np.array(bisector[0]))
        h_o = abs(np.dot(ad,bi_v)/np.linalg.norm(bi_v))

        # If the vector ad is perpendicular to the bisector then
        if np.allclose(h_o,0.0):
            h_o = abs(np.linalg.norm(np.array(b)-np.array(a)))

        # Here we'll use a bisection approach to find the correct value
        # for h. Technically there is a closed form solution but it is
        # really ugly and this might honestly be faster.
        correct_h = False
        h_test = h_o/2.
        prev = h_o/2.
        
        def project(a,b):
            """Projects a onto b.
            
            Args:
                a (numpy array): The first vector.
                b (numpy array): The second vector.

            Returns:
                proj (numpy array): The projection.
            """

            return b * np.dot(a,b)/np.linalg.norm(b)

        count = 0
        while not correct_h and count < 100:
            test_v = bi_v*h_test
            new_d = a + project(test_v,ad)
            if np.allclose(new_d,a):
                new_d = a + ad*h_test

            new_c = b + project(test_v,bc)
            if np.allclose(new_c,b):
                new_c = b + bc*h_test

            if abs(self._find_area(segments=self._find_segments(verts=[a,b,new_c,new_d]))-target) < self._eps:
                correct_h = True

            elif self._find_area(segments=self._find_segments(verts=[a,b,new_c,new_d])) > target:
                h_test -= prev/2.
                prev = prev/2.
            else:
                h_test +=prev/2.
                prev = prev/2.
            count += 1

        if count == 100:
            raise RuntimeError("Could not find correct cut line for trapezoid in 100 iterations.")

        if rest_of_poly is None:
            poly = [a,b,list(new_c),list(new_d)]
        else:
            poly = []
            for seg in rest_of_poly:
                if seg[0] == a:
                    poly.append(a)
                    poly.append(b)
                    poly.append(list(new_c))
                    poly.append(list(new_d))
                elif seg[0] == b:
                    poly.append(b)
                    poly.append(list(new_c))
                    poly.append(list(new_d))
                    poly.append(a)
                else:
                    poly.append(seg[0])

        if abs(self._find_area(segments=self._find_segments(verts=poly))-total_area) > self._eps:
            raise RuntimeError("Failed to find a cut line for the target area in "
                               "trapeziod_cut.")

        return poly            
        
    def _is_between(self,a,b,c):
        """Determines if point c is between points a and b.

        Args:
            a (numpy array): An endpoint of the line.
            b (numpy array): The other endpoint of the line.
            c (numpy array): A point in space.
        
        Returns:
            between (bool): True if c is on the line between a and b.
        """

        if not isinstance(a,np.ndarray):
            a = np.array(a)
        if not isinstance(b,np.ndarray):
            b = np.array(b)
        if not isinstance(c,np.ndarray):
            c = np.array(c)

        cross = np.cross(b-a,c-a)
        if abs(cross) > self._eps:
            return False

        dot = np.dot(b-a,c-a)
        if dot < 0:
            return False

        sqrlen = np.dot((b-a),(b-a))
        if dot > sqrlen:
            return False

        return True
    
    @staticmethod
    def _line_intersection(line1,line2):
        """Finds the intersection of two lines. Code contributed by Paul Draper:
        https://stackoverflow.com/questions/20677795/how-do-i-compute-the-intersection-point-of-two-lines-in-python
        
        Args:
            line1 (list): The end points of the first line.
            line2 (list): The end points of the second line.

        Returns:
            intersection (list): The intersection of the two lines.
        """
        
        xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
        ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1]) #Typo was here

        def det(a, b):
            return a[0] * b[1] - a[1] * b[0]

        div = det(xdiff, ydiff)
        if div == 0:
            x = (line1[0][0] + line2[1][0])/2.
            y = (line1[0][1] + line2[1][1])/2.
        else:
            d = (det(*line1), det(*line2))
            x = det(d, xdiff) / div
            y = det(d, ydiff) / div
            
        return [x, y]

    def _projections(self,line1, line2, bisection):
        """Finds the projections of the vertices of line1 onto line2 and vice
        versa such that the line between the vertex and the projection
        is perpendicular to the bisection line.

        Args:
            line1 (numpy array): The vertices of the first line.
            line2 (numpy array): The vertices of the second line.
            bisection (numpy array): The vertices of the bisection found by _angle_bisection

        Returns:
            projections (list): The projected point paired with the point it was projecting.
        """

        projections = []

        lines = [line1,line2]
        bisec = self._unit_vec(bisection[1],bisection[0])

        for i in range(2):
            for j in range(2):
                v = lines[i][j] - bisection[0]
                proj = bisec * np.dot(v,bisec)/np.linalg.norm(bisec)
                if i == 0:
                    tt = 1
                else:
                    tt = 0

                if not np.allclose(proj+bisection[0],lines[i][j]):
                    test_proj = self._line_intersection(lines[tt],[lines[i][j],list(proj+bisection[0])])
                    if np.allclose(0,test_proj[0]):
                        test_proj[0] = 0
                    if np.allclose(0,test_proj[1]):
                        test_proj[1] = 0
                    if self._is_between(lines[tt][0],lines[tt][1],test_proj):
                        projections.append([test_proj,lines[i][j]])

        return projections
        
    def _angle_bisection(self,line1,line2):

        """Finds the points that bisect the edges diffined by line1 and line2.

        Args:
            line1 (list): The endpoints of the first line.
            line2 (list): The endpoints of the second line.

        Returns:
            bisector (list): The endpoints of the bisector.
        """

        B = np.array(self._line_intersection(line1,line2))
        if not np.allclose(B,line1[0]):
            A = np.array(line1[0])
        else:
            A = np.array(line1[1])
        if not np.allclose(B,line2[1]):
            C = np.array(line2[1])
        else:
            C = np.array(line2[0])

        if self._is_between(A,C,B):
            if not np.allclose(B,line1[1]):
                A = np.array(line1[1])
            else:
                A = np.array(line1[0])
            if not np.allclose(B,line2[0]):
                C = np.array(line2[0])
            else:
                C = np.array(line2[1])
                
            if self._is_between(A,C,B):
                raise RuntimeError("Could not find a valid bisection of the line segmentns selected.")
            
        BA = self._unit_vec(B,A)
        BC = self._unit_vec(B,C)
        # The vector that bisects the angle between BA and BC is (BA+BC)/2
        bis_vec = (BA+BC)/2

        # we return B since it is the vertex of the bisection and
        # bis_vec+B since it is a second point on the vector starting at B.
        return [B,bis_vec+B]
    
    @staticmethod
    def _unit_vec(A,B):
        """Finds the unit vector that points from A to B.

        Args:
            A (numpy array): The starting point.
            B (numpy array): The ending point.

        Returns:
            uV (numpy array): The unit vector tha points from A to B.
        """

        A = np.array(A)
        B = np.array(B)
        dist = np.linalg.norm(B-A)
        return (B-A) / dist
    
    def _find_area(self, segments=None):
        """Finds the area of the polygon from the shoelace algorithm. Code
        contributed by Darius Bacon:
        https://stackoverflow.com/questions/451426/how-do-i-calculate-the-area-of-a-2d-polygon.

        Args:
            segments (list optional): A list of the line segments for the area. If None then 
              self._segments is used.
        
        Returns:
            area (float): The area of the polygon.
        """
        if segments is None:
            return 0.5 * abs(sum(x0*y1 - x1*y0
                                 for ((x0, y0), (x1, y1)) in self._segments))
        else:
            return 0.5 * abs(sum(x0*y1 - x1*y0
                                 for ((x0, y0), (x1, y1)) in segments))

    def _find_segments(self, verts=None):
        """Finds the segments of the polygon from the vertices.
        
        Args:
            verts (list, optional): A list of vertices, if none are supplied then the 
              self.vertices is used.

        Returns:
            segments (list): The paired segments of the vertices.
        """
        if verts is None:
            return list(zip(self.vertices, self.vertices[1:] + [self.vertices[0]]))
        else:
            return list(zip(verts, verts[1:] + [verts[0]]))

    def _find_permimter(self,segments=None):
        """Finds the perimiter of the polygon.

        Args:
            segments (list optional): A list of the line segments for the area. If None then 
              self._segments is used.

        Returns:
            perimiter (float): The perimiter.
        """
        if segments is None:
            return sum(np.sqrt((x1-x0)**2 + (y1-y0)**2) for ((x0,y0),(x1,y1)) in self._segments)
        else:
            return sum(np.sqrt((x1-x0)**2 + (y1-y0)**2) for ((x0,y0),(x1,y1)) in segments)
