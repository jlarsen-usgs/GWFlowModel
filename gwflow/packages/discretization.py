import numpy as np
from .package import PakBase


class Discretization(PakBase):
    """
    Package to define the Discretization of a Structured Groundwater Flow model


    Parameters
    ----------
    parent : GroundwaterFlow
        groundwater flow model instance
    nlay : int
        number of model layers
    nrow : int
        number of model rows
    ncol : int
        number of model columns
    delx : np.ndarray
        cell widths in the x direction. Array has shape ncol
    dely : np.ndarray
        cell heights in the y direction. Array has shape nrow
    top : np.ndarray
        model top elevation array. array should be sized (nrow, ncol)
    bottom : np.ndarray
        bottom elevation array. Array should have bottom elevations for every layer
        and be sized (nlay, nrow, ncol)
    isactive : np.ndarray
        model array that specifies if a cell is active (value>0) or inactive (value=0).
        array is sized (nlay, nrow, ncol)
    package_name : str
        user provided package name, default is "dis"
    """
    def __init__(
        self,
        parent,
        nlay,
        nrow,
        ncol,
        delx,
        dely,
        top,
        bottom,
        isactive,
        package_name="dis"
    ):
        super().__init__(parent, package_name)
        self._nlay = nlay
        self._nrow = nrow
        self._ncol = ncol
        self._delx = delx
        self._dely = dely
        self._top = np.ravel(top)
        self._botm = np.ravel(bottom)
        self._isactive = np.ravel(isactive)
        self._vertices = None
        self._iverts = None
        self._xycenters = None
        self._zcenters = None
        self._hvwa = None
        self._horiz_n2n_dist = None
        self._vert_n2n_dist = None
        self._area = None
        self._neighbors = None
        self._vneighbors = None

    def _calculate_cell_vertices(self):
        """
        Internal method to calculate vertices and iverts from delx and dely information
        """
        xv0 = np.tile([0] + list(np.add.accumulate(self._delx)), self._nrow + 1)
        yv0 = [0] + list(np.add.accumulate(self._dely))
        yv0 = np.repeat(yv0[::-1], self._ncol + 1)
        xyv = np.array([list(i) for i in zip(xv0, yv0)])

        iverts = []
        for row in range(self._nrow):
            for col in range(self._ncol):
                niv = row * (self._ncol + 1) + col
                iv = [niv, niv + 1, niv + self._ncol + 2, niv + self._ncol + 1]
                iverts.append(iv)

        iverts = np.array(iverts)
        self._iverts = iverts
        self._vertices = xyv

    @property
    def ncpl(self):
        return self._nrow * self._ncol

    @property
    def nnodes(self):
        return self._nlay * self._nrow * self._ncol

    @property
    def xyvertices(self):
        """
        Method to get xy vertices for each vertex in the model

        Returns
        -------
        np.array [(x_0, y_0) ... (x_n, y_n)]
        """
        if self._vertices is None:
            self._calculate_cell_vertices()
        return self._vertices

    @property
    def xycenters(self):
        """
        Method to get the xy center coordinates for each cell in model

        Returns
        -------
        np.ndarray [(x_0, y_0) ... (x_n, y_n)]
        """
        if self._xycenters is None:
            xyc = []
            for ivert in self.iverts:
                verts = self.xyvertices[ivert]
                xy = np.mean(verts, axis=0)
                xyc.append(xy)
            self._xycenters = np.array(xyc)
        return self._xycenters

    @property
    def zcenters(self):
        """
        Method to get the z-center coordinate for every cell in the model

        Returns
        -------
            np.ndarray [z0, ..., zn]
        """
        return np.mean([self.tops, self.bottoms], axis=0)

    @property
    def neighbors(self):
        """
        Method to get an array of neighbors for each node

        Returns
        -------
        np.ndarray (ncpl, 4) of neighbors. -1 indicates no neighbor in that position.

        Index positions are as follows:
        0 : column up
        1 : row left
        2 : row right
        3 : column down

        """
        # todo: I think we can do an isactive filter/trap in here...
        if self._neighbors is None:
            narr = np.full((self.nnodes, 4), -1)

            edges = []
            nodes = []
            for node, ivert in enumerate(self.iverts):
                for ix, iv in enumerate(ivert):
                    edges.append(tuple(sorted((ivert[ix - 1], iv))))
                    nodes.append(node)

            edge_dict = {}
            for ix, edge in enumerate(edges):
                node = nodes[ix]
                if edge in edge_dict:
                    edge_dict[edge].add(node)
                else:
                    edge_dict[edge] = {node, }

            neighbors = {i: set() for i in range(self.ncpl)}
            for _, nodes in edge_dict.items():
                for node in nodes:
                    for nd in nodes:
                        if node == nd:
                            continue
                        neighbors[node].add(nd)

            neighbors = {n: list(nn) for n, nn in neighbors.items()}
            for node, neighs in neighbors.items():
                for lay in range(self._nlay):
                    for n in neighs:
                        offset = n - node
                        nnode = node + (lay * self.ncpl)
                        nn = n + (lay * self.ncpl)
                        if offset == -self._ncol:
                            narr[nnode, 0] = nn
                        elif offset == -1:
                            narr[nnode, 1] = nn
                        elif offset == 1:
                            narr[nnode, 2] = nn
                        elif offset == self._ncol:
                            narr[nnode, 3] = nn
                        else:
                            raise Exception()

            self._neighbors = narr

        return self._neighbors

    @property
    def vertical_neighbors(self):
        """
        Get the vertical neighbor connectivity of cells in the model

        Returns
        -------
            np.ndarray (nnodes, 2) of neighbors. -1 indicates no neighbor in that position.

        Index positions are as follows:
        0 : layer up
        1 : layer down
        """
        if self._vneighbors is None:
            narr = np.full((self.nnodes, 2), -1)
            if self._nlay > 1:
                for node in range(self.nnodes):
                    if node - self.ncpl < 0:
                        pass
                    else:
                        narr[node, 0] = node - self.ncpl

                    if node + self.ncpl >= (self.nnodes):
                        pass
                    else:
                        narr[node, 1] = node + self.ncpl

            self._vneighbors = narr

        return self._vneighbors

    @property
    def hvwa(self):
        """
        Method to get the horizontal with (face width) of connections with neighboring
        cells

        Returns
        -------
        np.ndarray (ncpl, 4) of hvwa.

        Inedx positions are as follows and correspond to the neighbor array:
        0 : column up
        1 : row left
        2 : row right
        3 : column down

        """
        if self._hvwa is None:
            hvwa = np.full(self.neighbors.shape, np.nan)
            for node, neigh in enumerate(self.neighbors):
                ixnode = node
                while ixnode >= self.ncpl:
                    ixnode -= self.ncpl

                node_edges = []
                ivrts = self.iverts[ixnode]
                for ix, iv in enumerate(ivrts):
                    node_edges.append(tuple(sorted([ivrts[ix - 1], iv])))
                for iix, nd in enumerate(neigh):
                    if nd == -1:
                        continue

                    ixnd = nd
                    while ixnd >= self.ncpl:
                        ixnd -= self.ncpl

                    civ = None
                    nivrts = self.iverts[ixnd]
                    for ix, iv in enumerate(nivrts):
                        tmp = tuple(sorted([nivrts[ix - 1], iv]))
                        if tmp in node_edges:
                            civ = tmp
                            break

                    if civ is None:
                        raise AssertionError()

                    x0, y0 = self.xyvertices[civ[0]]
                    x1, y1 = self.xyvertices[civ[1]]
                    hvwa[node, iix] = self.distance_eq(x0, y0, x1, y1)

            self._hvwa = hvwa

        return self._hvwa

    @property
    def horizontal_node_to_node_distance(self):
        """
        Get the horizontal cell center to cell center distance for each neighbor

        Returns
        -------
        np.ndarray (ncpl, 4) of n2n_dist

        Index positions are as follows and correspond to the neighbor array:
        0 : column up
        1 : row left
        2 : row right
        3 : column down

        """
        if self._horiz_n2n_dist is None:
            n2n_dist = np.full(self.neighbors.shape, np.nan)
            for node, neighs in enumerate(self.neighbors):
                ixnode = node
                while ixnode >= self.ncpl:
                    ixnode -= self.ncpl

                x0, y0 = self.xycenters[ixnode]
                for ix, n in enumerate(neighs):
                    if n == -1:
                        continue
                    ixn = n
                    while ixn >= self.ncpl:
                        ixn -= self.ncpl

                    x1, y1 = self.xycenters[ixn]
                    n2n_dist[node, ix] = self.distance_eq(x0, y0, x1, y1)

            self._horiz_n2n_dist = n2n_dist

        return self._horiz_n2n_dist

    @property
    def vertical_node_to_node_distance(self):
        """
        Get the vertical cell center to cell center distance for each neighbor

        Returns
        -------
        np.ndarray (nnodes, 2) or n2n_dist

        Index positions are as follows and correspond to the vertical neighbor array:
        0 : layer up
        1 : layer down

        """
        if self._vert_n2n_dist is None:
            n2n_dist = np.full(self.vertical_neighbors.shape, np.nan)
            for node, neighs in enumerate(self.vertical_neighbors):
                z0 = self.zcenters[node]
                for ix, n in enumerate(neighs):
                    if n == -1:
                        continue
                    z1 = self.zcenters[n]
                    n2n_dist[node, ix] = np.abs(z0 - z1)

            self._vert_n2n_dist = n2n_dist

        return self._vert_n2n_dist

    @property
    def iverts(self):
        """
        Method to get the iverts that compose each cell

        Returns
        -------
        np.array [(iv_0 ... iv_x)_0 .... (iv_0 ... iv_x)_n]
        """
        if self._iverts is None:
            self._calculate_cell_vertices()
        return self._iverts

    @property
    def tops(self):
        """
        Method to get the top elevation of every node in the model

        Returns
        -------
        np.array (size nnodes)
        """
        if self._botm.size != self._top.size:
            i0 = 0
            top = self._top
            for lay in range(1, self._nlay):
                i1 = self.ncpl * lay
                top = np.append(top, self._botm[i0: i1])
                i0 += self.ncpl
            self._top = top
        return self._top

    @property
    def bottoms(self):
        """
        Method to get the botm elevation of every node in the model

        Returns
        -------
        np.array (size nnodes)
        """
        return self._botm

    @property
    def cell_thick(self):
        """
        Method to get the cell thickness of every node in the model

        Returns
        -------
        np.array (size nnodes)
        """
        return self.tops - self.bottoms

    @property
    def cell_area(self):
        """
        Method to get the area of a cell

        Returns
        -------
            np.ndarray (number of nodes)
        """
        if self._area is None:
            xverts = np.full(self.iverts.shape, np.nan)
            yverts = np.full(self.iverts.shape, np.nan)
            for ix, ivts in enumerate(self.iverts):
                xverts[ix] = self.xyvertices[list(ivts), [0, 0, 0, 0]]
                yverts[ix] = self.xyvertices[list(ivts), [1, 1, 1, 1]]

            area_x2 = np.zeros((1, len(xverts)))
            for i in range(xverts.shape[-1]):
                # calculate the determinant of each line in polygon
                area_x2 += xverts[:, i - 1] * yverts[:, i] - yverts[:, i - 1] * xverts[:, i]

            area = np.abs(area_x2 / 2.0)
            self._area = np.ravel(area)
        return self._area

    @property
    def cell_volume(self):
        """
        Returns the volume of each node in the model

        Returns
        -------
        np.ndarray
        """
        return self.cell_area * self.cell_thick

    def saturated_thick(self, heads):
        """
        Calculates the saturated thickness of each node

        Parameters
        ----------
        heads : np.ndarray

        Returns
        -------
        np.ndarray
        """
        heads = np.ravel(heads)
        sat_thick = heads - self.bottoms
        sat_thick = np.where(sat_thick < 0, 0, sat_thick)
        return sat_thick

    @staticmethod
    def distance_eq(x0, y0, x1, y1):
        """
        Standard distance equation calculation

        Parameters
        ----------
        x0 : float
            x-coordinate 0
        y0 : float
            y-coordiante 0
        x1 : float
            x-coordinate 1
        y1 : float
            y-coordinate 1

        Returns
        -------
            dist : float
        """
        a2 = (x0 - x1) ** 2
        b2 = (y0 - y1) ** 2
        dist = np.sqrt(a2 + b2)
        return dist

    @staticmethod
    def package_type():
        """
        Returns the specific package type acronym
        """
        return "DIS"