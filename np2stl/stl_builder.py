"""This module provides the STLBuilder class.

"""
import numpy as np
from stl import mesh
from tqdm import tqdm


class STLBuilder(object):
    """Class accepting data from numpy array or values on a grid.
    Produces STL files from that data.

    """

    def __init__(self):
        """Set up the class.

        Creates empty list of vertices and faces

        """
        self.__reset()

    def __reset(self):
        """Reset all vertices and faces.

        """
        self.vertices = []
        self.faces = []

    def __clean_up(self):
        """Clean up the vertices.

        Remove duplicate vertices.

        """
        self.__remove_duplicate_vertices()

    def __remove_duplicate_vertices(self):
        """Delete duplicate vertices, change mapping in faces.

        """
        self.vertices = np.array(self.vertices)
        self.faces = np.array(self.faces)

        cleaned_vertices = []
        # mapping maps old id to new id
        vertex_mapping = np.full(len(self.vertices), -1)

        for vertex_id, vertex in tqdm(enumerate(self.vertices),
                                      total=len(self.vertices),
                                      ascii=True,
                                      desc="Removing duplicate vertices",
                                      unit="vertex"):
            if vertex_mapping[vertex_id] != -1:  # we have already worked on this vertex
                continue
            # we add the vertex to the cleaned vertices
            cleaned_vertices.append(vertex)
            new_vertex_id = len(cleaned_vertices) - 1
            # there can be multiple occasions of this vertex
            indices = np.unique(np.where(np.all(self.vertices == vertex, axis=-1))[0])
            # fill out the mapping
            vertex_mapping[indices] = new_vertex_id

        new_faces = np.zeros_like(self.faces)
        # now apply the new indices
        for face_id, face in enumerate(self.faces):
            new_faces[face_id] = vertex_mapping[face]
        self.faces = new_faces
        self.vertices = np.array(cleaned_vertices)

    def save(self,
             filename,
             cleanup=False):
        """Save the 3d object to a binary STL file

        Parameters
        ----------
        filename : str
            Path to store the file at.
        cleanup : bool, optional
            Whether to clean up the vertices before saving. Default False.

        """
        # prepare for building the stl file
        # everything is nice when a numpy array
        self.vertices = np.array(self.vertices)
        self.faces = np.array(self.faces)

        if cleanup:
            self.__clean_up()

        # create the mesh
        my_mesh = mesh.Mesh(np.zeros(self.faces.shape[0], dtype=mesh.Mesh.dtype))
        for face_id, face in enumerate(self.faces):
            for vertex_id in range(3):
                my_mesh.vectors[face_id][vertex_id] = self.vertices[face[vertex_id]]

        if filename.split(".")[-1] != "stl":
            filename += ".stl"

        my_mesh.save(filename)

    def _add_triangle(self,
                      points):
        """Add a single triangle to the 3d object.

        Parameters
        ----------
        points : iterable of 3 3-vectors
            Corners of the triangles.

        """
        assert len(points) == 3
        # we have two traingles to draw
        self.vertices.extend(points)
        # we need the actual position of the last vertices for reference for the faces
        n_vertices = len(self.vertices)
        self.faces.append([n_vertices - 3, n_vertices - 2, n_vertices - 1])

    def _add_flat_rectangle(self, points):
        """Add a flat rectangle to the 3d object.

        Parameters
        ----------
        points : iterable of 4 3-vectors
            Corners of the rectangles.

        """
        assert len(points) == 4
        self._add_triangle(points[:-1])
        self._add_triangle(points[1:])

    def _add_base(self,
                  base_height=0.02,
                  xmin=0.,
                  xmax=1.,
                  ymin=0.,
                  ymax=1.):
        """Add a baseplate and brim to the 3d object.

        The baseplate serves as a foundation of the 3d object.
        The brim creates the connection to the rest of the 3d object.

        Parameters
        ----------
        base_height : float
            Height of the baseplate as fraction of the longest side. Default 0.05.
        xmin : float
            Lower bound in x-direction for the baseplate. Default 0.
        xmax : float
            Upper bound in x-direction for the baseplate. Default 1.
        ymin : float
            Lower bound in y-direction for the baseplate. Default 0.
        ymax : float
            Upper bound in y-direction for the baseplate. Default 1.

        """
        self._add_baseplate(baseplate_height=base_height,
                            xmin=xmin,
                            xmax=xmax,
                            ymin=ymin,
                            ymax=ymax)
        self._add_brims(baseplate_height=base_height,
                        xmin=xmin,
                        xmax=xmax,
                        ymin=ymin,
                        ymax=ymax)

    def _add_baseplate(self,
                       baseplate_height=0.02,
                       xmin=0,
                       xmax=1,
                       ymin=0,
                       ymax=1):
        """Add a baseplate to the 3d object.

        The baseplate serves as a foundation of the 3d object.

        Parameters
        ----------
        base_height : float
            Height of the baseplate as fraction of the longest side. Default 0.05.
        xmin : float
            Lower bound in x-direction for the baseplate. Default 0.
        xmax : float
            Upper bound in x-direction for the baseplate. Default 1.
        ymin : float
            Lower bound in y-direction for the baseplate. Default 0.
        ymax : float
            Upper bound in y-direction for the baseplate. Default 1.

        """
        self._add_flat_rectangle([[xmin, ymin, -baseplate_height],
                                  [xmin, ymax, -baseplate_height],
                                  [xmax, ymin, -baseplate_height],
                                  [xmax, ymax, -baseplate_height]])

    def _add_brims(self,
                   baseplate_height=0.02,
                   xmin=0,
                   xmax=1,
                   ymin=0,
                   ymax=1):
        """Add brims to the 3d object.

        The brims connects the baseplate with the rest of the 3d object.

        Parameters
        ----------
        base_height : float
            Height of the baseplate as fraction of the longest side. Default 0.05.
        xmin : float
            Lower bound in x-direction for the baseplate. Default 0.
        xmax : float
            Upper bound in x-direction for the baseplate. Default 1.
        ymin : float
            Lower bound in y-direction for the baseplate. Default 0.
        ymax : float
            Upper bound in y-direction for the baseplate. Default 1.

        """
        face_0 = [[xmin, ymin, -baseplate_height],
                  [xmin, ymin, 0],
                  [xmax, ymin, -baseplate_height],
                  [xmax, ymin, 0]]
        face_1 = [[xmin, ymin, -baseplate_height],
                  [xmin, ymin, 0],
                  [xmin, ymax, -baseplate_height],
                  [xmin, ymax, 0]]
        face_2 = [[xmax, ymax, -baseplate_height],
                  [xmax, ymax, 0],
                  [xmax, ymin, -baseplate_height],
                  [xmax, ymin, 0]]
        face_3 = [[xmax, ymax, -baseplate_height],
                  [xmax, ymax, 0],
                  [xmin, ymax, -baseplate_height],
                  [xmin, ymax, 0]]
        self._add_flat_rectangle(face_0)
        self._add_flat_rectangle(face_1)
        self._add_flat_rectangle(face_2)
        self._add_flat_rectangle(face_3)

    def from_numpy_2dhist(self,
                          hist,
                          bin_edges_x,
                          bin_edges_y,
                          style="pyramid",
                          log=False,
                          aspect_ratio=1.33):
        """Create a 3d object from a numpy histogram.

        Parameters
        ----------
        hist : 2D numpy array
            The bin entries.
        bin_edges_x : iterable of float
            The bin edges in x direction of the histogram.
        bin_edges_y : iterable of float
            The bin edges in y direction of the histogram.
        style : str, optional
            The style to produce the 3d object in. At the moment only 'pyramid' is supported. Default 'pyramid'.
        log : bool, optional
            Whether to rescale the values of `hist` to a logarithmic axis. Default False.
        aspect_ratio : float, optional
            The aspect ratio in x-y to use for the 3d object.
            The x scale will be longer by this factor than the y scale.
            Default 1.33.

        Raises
        ------
        ValueError
            If an invalid object style was provided.

        """
        if style == "pyramid":
            self._pyramid_from_numpy_2dhist(hist, bin_edges_x, bin_edges_y, log=log, aspect_ratio=aspect_ratio)
        else:
            raise ValueError("Invalid style <{style}>".format(style=style))

    def from_gridpoints(self,
                        values,
                        x_values,
                        y_values,
                        style="pyramid",
                        log=False,
                        aspect_ratio=1.33):
        """Create a 3d object from points on a regular 2d grid.

        Parameters
        ----------
        values : 2D numpy array
            The z-values of the points.
        x_values : iterable of float
            The x coordinates of the points on the grid.
        y_values : iterable of float
            The y coordinates of the points on the grid.
        style : str, optional
            The style to produce the 3d object in. At the moment only 'pyramid' is supported. Default 'pyramid'.
        log : bool, optional
            Whether to rescale the values of `hist` to a logarithmic axis. Default False.
        aspect_ratio : float, optional
            The aspect ratio in x-y to use for the 3d object.
            The x scale will be longer by this factor than the y scale.
            Default 1.33.

        Raises
        ------
        ValueError
            If an invalid object style was provided.

        """
        if style == "pyramid":
            self._pyramid_from_gridpoints(values, x_values, y_values, log=log, aspect_ratio=aspect_ratio)
        else:
            raise ValueError("Ivalid style <{style}>".format(style=style))

    def _pyramid_from_numpy_2dhist(self,
                                   hist,
                                   bin_edges_x,
                                   bin_edges_y,
                                   log=False,
                                   aspect_ratio=1.33):
        """Create a pyramid-style 3d object from a numpy histogram.

        Parameters
        ----------
        hist : 2D numpy array
            The bin entries.
        bin_edges_x : iterable of float
            The bin edges in x direction of the histogram.
        bin_edges_y : iterable of float
            The bin edges in y direction of the histogram.
        log : bool, optional
            Whether to rescale the values of `hist` to a logarithmic axis. Default False.
        aspect_ratio : float, optional
            The aspect ratio in x-y to use for the 3d object.
            The x scale will be longer by this factor than the y scale.
            Default 1.33.

        """
        # make sure everything is a numpy array
        hist = np.array(hist)
        bin_edges_x = np.array(bin_edges_x)
        bin_edges_y = np.array(bin_edges_y)

        # in pyramid mode, we need to add zero value at the edges, we add additional points at each side
        midpoints_x = 0.5 * (bin_edges_x[:-1] + bin_edges_x[1:])
        midpoints_y = 0.5 * (bin_edges_y[:-1] + bin_edges_y[1:])
        self._pyramid_from_gridpoints(hist, midpoints_x, midpoints_y, log=log, aspect_ratio=aspect_ratio)

    def _pyramid_from_gridpoints(self,
                                 values,
                                 x_values,
                                 y_values,
                                 log=False,
                                 aspect_ratio=1.33):
        """Create a pyramid-style 3d object from points on a regular 2d grid.

        Parameters
        ----------
        values : 2D numpy array
            The z-values of the points.
        x_values : iterable of float
            The x coordinates of the points on the grid.
        y_values : iterable of float
            The y coordinates of the points on the grid.
        style : str, optional
            The style to produce the 3d object in. At the moment only 'pyramid' is supported. Default 'pyramid'.
        log : bool, optional
            Whether to rescale the values of `hist` to a logarithmic axis. Default False.
        aspect_ratio : float, optional
            The aspect ratio in x-y to use for the 3d object.
            The x scale will be longer by this factor than the y scale.
            Default 1.33.

        Raises
        ------
        ValueError
            If an invalid object style was provided.

        """
        self.__reset()
        samples_x = np.concatenate([[2 * x_values[0] - x_values[1]], x_values, [2 * x_values[-1] - x_values[-2]]])
        samples_y = np.concatenate([[2 * y_values[0] - y_values[1]], y_values, [2 * y_values[-1] - y_values[-2]]])
        # we rescale the x scale with the aspect ratio
        samples_x = (samples_x - samples_x[0]) / (samples_x[-1] - samples_x[0]) * aspect_ratio
        # we rescale the y scale to (0, 1)
        samples_y = (samples_y - samples_y[0]) / (samples_y[-1] - samples_y[0])

        # we can take care of the values now
        # take the log if we need to
        if log:
            values = np.log10(np.clip(values, np.min(values[np.nonzero(values)]), None))
        # rescale to [0, 1]
        hist_min = np.min(values)
        hist_max = np.max(values)
        heights = (values - hist_min) / (hist_max - hist_min) * 0.5
        # now we can safely pad with the minimum value zero
        heights = np.pad(heights, 1, constant_values=[0])

        self._add_base(xmin=0, xmax=samples_x[-1], ymin=0, ymax=samples_y[-1])

        # start with the actual data
        # we do not care about duplicate vertices for now
        for x_id, x_value in enumerate(samples_x[:-1]):
            for y_id, y_value in enumerate(samples_y[:-1]):
                # we have two traingles to draw
                triag_1 = [[x_value, y_value, heights[x_id, y_id]],
                           [samples_x[x_id + 1], y_value, heights[x_id + 1, y_id]],
                           [samples_x[x_id + 1], samples_y[y_id + 1], heights[x_id + 1, y_id + 1]]]
                self._add_triangle(triag_1)
                triag_1 = [[x_value, y_value, heights[x_id, y_id]],
                           [x_value, samples_y[y_id + 1], heights[x_id, y_id + 1]],
                           [samples_x[x_id + 1], samples_y[y_id + 1], heights[x_id + 1, y_id + 1]]]
                self._add_triangle(triag_1)
