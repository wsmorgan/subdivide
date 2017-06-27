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

        Raises:
            RuntimeError: if the code finds more than 3 sub-areas which should 
                be impossible.
            RuntimeError: if the code cannot reconstruct the remaining polygon 
                after removing the sub_polygon's vertices.
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
                    if len(divisions) == 1 and self._find_area(segments=divisions[0]) > area:
                        if len(divisions[0]) == 3:
                            test_poly = self._triangle_cut(divisions[0],area,segments,None)
                        elif len(divisions[0]) == 4:
                            test_poly = self._trapezoid_cut(divisions[0],area,bisector,segments,None)
                    elif len(divisions) == 2:
                        if self._find_area(segments=divisions[0]) > area:
                            if len(divisions[0]) == 3:
                                test_poly = self._triangle_cut(divisions[0],area,segments,None)
                            elif len(divisions[0]) == 4:
                                test_poly = self._trapezoid_cut(divisions[0],area,bisector,segments,None)
                        if self._find_area(segments=divisions[1]) > area and test_poly is None:
                            if len(divisions[1]) == 3:
                                test_poly = self._triangle_cut(divisions[1],area,segments,None)
                            elif len(divisions[1]) == 4:
                                test_poly = self._trapezoid_cut(divisions[1],area,bisector,segments,None)
                    elif len(divisions) == 3:
                        area1 = self._find_area(segments=divisions[0])
                        area2 = self._find_area(segments=divisions[1])
                        area3 = self._find_area(segments=divisions[2])
                        if area1 > area:
                            if len(divisions[0]) == 3:
                                test_poly = self._triangle_cut(divisions[0],area,segments,None)
                            elif len(divisions[0]) == 4:
                                test_poly = self._trapezoid_cut(divisions[0],area,bisector,segments,None)
                        elif (area1 + area2) > area:
                            if len(divisions[1]) == 3:
                                test_poly = self._triangle_cut(divisions[1],area,segments,divisions[0])
                            elif len(divisions[1]) == 4:
                                test_poly = self._trapezoid_cut(divisions[1],area,bisector,segments,divisions[0])
                        if area3 > area and test_poly is None:
                            if len(divisions[2]) == 3:
                                test_poly = self._triangle_cut(divisions[2],area,segments,None)
                            elif len(divisions[2]) == 4:
                                test_poly = self._trapezoid_cut(divisions[2],area,bisector,segments,None)
                    else: #pragma: no cover
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
                #this is a new vertex that splits the regoins.
                remaining_poly.append(vert)
            elif vert in verts and vert not in sub_poly:
                # This is a vertex in the original polygon only.
                remaining_poly.append(vert)

        # Check to make sure we didn't miss any vertices. This will
        # only happen when the sub_poly and remaining polygon to share
        # a vertex of the original polygon.
        test_area = self._find_area(segments=self._find_segments(verts=remaining_poly))
        if not np.allclose(test_area+area,total_area):
            area_check = False
            for vert in sub_poly:
                if vert not in remaining_poly:
                    test_case = remaining_poly + [vert]
                    temp_area = self._find_area(segments=self._find_segments(verts=test_case))
                    if np.allclose(temp_area+area,total_area):
                        remaining_poly.append(vert)
                        area_check = True
                        break

        else:
            area_check = True            

        if not area_check: #pragma: no cover
            raise RuntimeError("Couldn't find the vertices of the remaining polygon.")

        return self._counter_clockwise_sort(sub_poly), self._counter_clockwise_sort(remaining_poly)

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
        
        new_segments = []

        if len(new_points) == 2 and ((np.allclose(new_points[0][0],new_points[1][1]) and np.allclose(new_points[0][1],new_points[1][0])) or (np.allclose(new_points[0][0],new_points[1][0]) and np.allclose(new_points[0][1],new_points[1][1]))):
            new_points = [new_points[0]]

        for point in new_points:
            option1 = (point[1][0]-point[0][0])*(point[1][1]+point[0][1])
            if option1 < 0:
                new_segments.append((point[0],point[1]))
            else:
                new_segments.append((point[1],point[0]))

            # if the point isn't an end point then we need to find out
            # which line it's in to create new subsegments of the line.
            if not (np.allclose(point[0],line1[0]) or np.allclose(point[0],line1[1])) and not (np.allclose(point[0],line2[0]) or np.allclose(point[0],line2[1])):
                if self._is_between(line1[0],line1[1],point[0]):
                    new_segments.append((line1[0],point[0]))
                    new_segments.append((point[0],line1[1]))
                elif self._is_between(line2[0],line2[1],point[0]):
                    new_segments.append((line2[0],point[0]))
                    new_segments.append((point[0],line2[1]))

        return new_segments, bisector, new_points
    
    def _create_new_areas(self,new_segments,old_segments):
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

        new_segments_local = []
        for seg in new_segments:
            if seg not in new_segments_local:
                new_segments_local.append(seg)
            if seg[::-1] not in new_segments_local:
                new_segments_local.append(seg[::-1])

        all_segments = new_segments_local+old_segments

        new_areas = []
        verts_lists = []
        for seg in new_segments_local:
            new_path =[seg]
            new_verts = [seg[0]]
            cur_seg = seg
            for test_seg in all_segments:
                if not (np.allclose(test_seg[0],cur_seg[0]) and np.allclose(test_seg[1],cur_seg[1])) and not (np.allclose(test_seg[0],cur_seg[1]) and np.allclose(test_seg[1],cur_seg[0])) and np.allclose(test_seg[0],cur_seg[1]):
                    
                    between = False
                    for v in new_verts:
                        if self._is_between(test_seg[0],test_seg[1],v) and v not in test_seg:
                            between = True
                            break
                        
                    if not between:
                        new_path.append(test_seg)
                        new_verts.append(test_seg[0])
                        cur_seg = test_seg
                        
                        if np.allclose(cur_seg[1],seg[0]):
                            break

                        for test_seg_2 in new_segments_local:
                            if not (np.allclose(test_seg_2[0],cur_seg[0]) and np.allclose(test_seg_2[1],cur_seg[1])) and not (np.allclose(test_seg_2[0],cur_seg[1]) and np.allclose(test_seg_2[1],cur_seg[0])) and np.allclose(test_seg_2[0],cur_seg[1]):

                                between = False
                                for v in new_verts:
                                    if self._is_between(test_seg_2[0],test_seg_2[1],v) and v not in test_seg_2:
                                        between = True
                                        break
                        
                                if not between:
                                    new_path.append(test_seg_2)
                                    new_verts.append(test_seg_2[0])
                                    cur_seg = test_seg_2
                                    if np.allclose(cur_seg[1],seg[0]):
                                        break

                    if np.allclose(cur_seg[1],seg[0]):
                        break
                            
            if not np.allclose(cur_seg[1],seg[0]):
                for test_seg in all_segments:
                    if not (np.allclose(test_seg[0],cur_seg[0]) and np.allclose(test_seg[1],cur_seg[1])) and not (np.allclose(test_seg[0],cur_seg[1]) and np.allclose(test_seg[1],cur_seg[0])) and np.allclose(test_seg[0],cur_seg[1]):
                        between = False

                        for v in new_verts:
                            if self._is_between(test_seg[0],test_seg[1],v) and v not in test_seg:
                                between = True
                                break
                            
                        if not between:
                            new_path.append(test_seg)
                            new_verts.append(test_seg[0])
                            cur_seg = test_seg
                            if np.allclose(cur_seg[1],seg[0]):
                                break
                            
                            for test_seg_2 in new_segments_local:
                                if not (np.allclose(test_seg_2[0],cur_seg[0]) and np.allclose(test_seg_2[1],cur_seg[1])) and not (np.allclose(test_seg_2[0],cur_seg[1]) and np.allclose(test_seg_2[1],cur_seg[0])) and np.allclose(test_seg_2[0],cur_seg[1]):
                                    between = False
                                    
                                    for v in new_verts:
                                        if self._is_between(test_seg_2[0],test_seg_2[1],v) and v not in test_seg_2:
                                            between = True
                                            break
                        
                                    if not between:
                                        new_path.append(test_seg_2)
                                        new_verts.append(test_seg_2[0])
                                        cur_seg = test_seg_2
                                        if np.allclose(cur_seg[1],seg[0]):
                                            break

                        if np.allclose(cur_seg[1],seg[0]):
                            break
                        
            cur_verts = [x for (x,y) in new_path]
            cur_verts = self._counter_clockwise_sort(cur_verts)

            if not new_path in new_areas and len(new_path) > 2 and len(new_path) <= len(self._segments) and self._find_permimter(segments=new_path) <= self.perimiter and np.allclose(new_path[0][0],new_path[-1][1]) and cur_verts not in verts_lists:
                new_areas.append(new_path)
                verts_lists.append(cur_verts)

        return new_areas

    def _triangle_cut(self,segments,total_area,orig_segments,rest_of_poly):
        """Finds the desired cut inside a triangle to get the correct area.
        
        Args:
            segments (list): The line segments that form the triangel.
            total_area (float): The area desired after the cut.
            orig_segments (list): The line segments of the uncut polygon.
            rest_of_poly (list): The line segments containing the rest of the
                polygon whose area will contribute.

        Returns:
            poly (list): The vertices that form the polygon with the desired area.

        Raises:
            RunTimeError: A RunTimeError is raised if the correct area cannot be found.
            RunTimeError: A RunTiemError is raised if 1 of segments passed to 
                the subroutine is not part of the original polygon.
        """

        if rest_of_poly is None:
            target = total_area
        else:
            target = total_area - self._find_area(rest_of_poly)
            
        if segments[0] in orig_segments:
            b = segments[1][1]
            c = segments[0][0]
            a = segments[0][1]
        elif segments[1] in orig_segments:
            b = segments[2][1]
            a = segments[1][1]
            c = segments[1][0]
        elif segments[2] in orig_segments:
            b = segments[0][1]
            a = segments[2][0]
            c = segments[2][1]
        else:
            RuntimeError("Could not find a line segment from the original polygon in "
                         "_triangle_cut. This should not be possible.")

        cut_point = list(np.array(c) + target/self._find_area(segments=segments)*(np.array(b)-np.array(c)))

        between = False
        for seg in orig_segments:
            if self._is_between(seg[0],seg[1],cut_point):
                between = True
                break

        # If the cut point isn't on one of the original polygon edges
        # we got the order of a and c wrong.
        if not between:
            temp_a = a
            a = c
            c = temp_a
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
                    
        if abs(self._find_area(segments=self._find_segments(verts=poly))-total_area) > self._eps: #pragma: no cover
            raise RuntimeError("Failed to find a cut line for the target area in "
                               "triange_cut.")

        return poly

    def _trapezoid_cut(self,segments,total_area,bisector,orig_segments,rest_of_poly):
        """Finds the desired cut inside a trapezoid to get the correct area.
        
        Args:
            segments (list): The line segments that form the trapezoid.
            total_area (float): The area desired after the cut.
            bisector (list): The endpoints of the bisector.
            orig_segments (list): The line segments of the uncut polygon.
            rest_of_poly (list): The line segments containing the rest of the
                polygon whose area will contribute.

        Returns:
            poly (list): The vertices that form the polygon with the desired area.

        Raises:
            RunTimeError: A RunTimeError is raised if the corroct area cannot be found.
            RunTimeError: A RunTiemError is raised if fewer than 2 segments passed to 
                the subroutine are not part of the original polygon.
        """
        
        if rest_of_poly is None:
            target = total_area
        else:
            target = total_area - self._find_area(rest_of_poly)

        if segments[0] in orig_segments and segments[1] in orig_segments:
            a = segments[0][0]
            b = segments[1][0]
            c = segments[1][1]
            d = segments[2][1]
        elif segments[1] in orig_segments and segments[2] in orig_segments:
            a = segments[1][0]
            b = segments[2][0]
            c = segments[2][1]
            d = segments[3][1]
        elif segments[2] in orig_segments and segments[3] in orig_segments:
            a = segments[2][0]
            b = segments[3][0]
            c = segments[3][1]
            d = segments[0][1]
        elif segments[3] in orig_segments and segments[0] in orig_segments:
            a = segments[3][0]
            b = segments[0][0]
            c = segments[0][1]
            d = segments[1][1]
        else:
            RuntimeError("Could not find the correct segments in the _trapezoid_cut "
                         "routine. The trapezoid constructed does not have 2 sides from the "
                         "original shape. This should not be possileb.")

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
        h_test = 1./2.
        prev_step = 1./2.
        step = 1./2.
        
        def project(a,b):
            """Projects a onto b.
            
            Args:
                a (numpy array): The first vector.
                b (numpy array): The second vector.

            Returns:
                proj (numpy array): The projection.
            """

            return b * np.dot(a,b)/np.linalg.norm(b)

        test_v = bi_v*h_test
        new_d = a + project(test_v,ad)
        if not self._is_between(a,d,new_d):
            bi_v = self._unit_vec(np.array(bisector[0]),np.array(bisector[1]))
            
        count = 0
        prev_area = self._find_area(segments=self._find_segments(verts=[a,b,c,d]))
        while not correct_h and count < 100:
            test_v = bi_v*h_test
            new_d = a + project(test_v,ad)
            if np.allclose(new_d,a):
                new_d = a + ad*h_test

            new_c = b + project(test_v,bc)
            if np.allclose(new_c,b):
                new_c = b + bc*h_test

            cur_area = self._find_area(segments=self._find_segments(verts=[a,b,new_c,new_d]))
            if abs(cur_area-target) < self._eps:
                correct_h = True

            elif abs(cur_area-prev_area)>abs(cur_area-target):
                if cur_area > target:
                    prev_step = step
                    step = step/2.
                    h_test -= step
                    prev_area = cur_area
                else:
                    prev_step = step
                    step = step/2.
                    h_test +=step
                    prev_area = cur_area
            else:
                if cur_area > target:
                    step = prev_step
                    h_test -= step
                    prev_area = cur_area
                else:
                    step = prev_step
                    h_test +=step
                    prev_area = cur_area
                
            count += 1

        if count == 100: #pragma: no cover
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

        if abs(self._find_area(segments=self._find_segments(verts=poly))-total_area) > self._eps: #pragma: no cover
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
        ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1]) 

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
                    for vert in self.vertices:
                        if np.allclose(test_proj,vert):
                            test_proj = vert
                            break
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
                
            if self._is_between(A,C,B): #pragma: no cover
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

    @staticmethod
    def _find_centroid(points):
        """Finds the centroid of the given list of points. 

        Args:
            points (list): A list of [x, y] pairs.

        Returns:
            centroid (list): The [x, y] pair of the centroid.
        """

        x = [p[0] for p in points]
        y = [p[1] for p in points]
        n = len(points)
        
        centroid = [sum(x)/float(n),sum(y)/float(n)]

        return centroid

    def _counter_clockwise_sort(self,points):
        """Sorts the points to be in counter clockwise oreder.

        Args:
            points (list): A list of [x, y] pairs:

        Returns:
            cc_points (list): A list of the [x, y] pairs sorted to be 
                in counter clockwise order.
        """

        self._center = self._find_centroid(points)

        def clockwiseangle_and_distance(point):
            """Returns the angle and distance from the centeroid of the point.
            This code is modified from code contributed by MSeifert at:
            https://stackoverflow.com/questions/41855695/sorting-list-of-two-dimensional-coordinates-by-clockwise-angle-using-python

            Args:
                point (list): The [x,y] pair to be sorted.
            
            Returns:
                angle, distance (float, float): The angle and the distance from the centroid 
                    of the polygon.
            """
            import math

            refvec = [0,1]
            # Vector between point and the origin: v = p - 
            vector = [point[0]-self._center[0], point[1]-self._center[1]]
            # Length of vector: ||v||
            lenvector = math.hypot(vector[0], vector[1])
            # If length is zero there is no angle
            if lenvector == 0:
                return -math.pi, 0
            # Normalize vector: v/||v||
            normalized = [vector[0]/lenvector, vector[1]/lenvector]
            dotprod  = normalized[0]*refvec[0] + normalized[1]*refvec[1]     # x1*x2 + y1*y2
            diffprod = refvec[1]*normalized[0] - refvec[0]*normalized[1]     # x1*y2 - y1*x2
            angle = math.atan2(diffprod, dotprod)
            # Negative angles represent counter-clockwise angles so we need to subtract them 
            # from 2*pi (360 degrees)
            if angle < 0:
                return 2*math.pi+angle, lenvector
            # I return first the angle because that's the primary sorting criterium
            # but if two vectors have the same angle then the shorter distance should come first.
            return angle, lenvector

        cc_points = sorted(points,key=clockwiseangle_and_distance)

        return cc_points[::-1]
