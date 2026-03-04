import numpy as np
import scipy as sp
from scipy.sparse.linalg import gmres


class GroundwaterFlow:
    """
    Main groundwater flow model instance

    Parameters
    ----------
    modelname : str
        name of the model

    """
    def __init__(self, modelname):
        self.modelname = modelname
        self._dis = None
        self._hyd = None
        self._sto = None
        self._ic = None
        self._stress_pkgs = []
        self._chd_pkgs = []

        # solution stuff
        self._hold = None
        self._amat = None
        self._rhs = None
        self._hcof = None
        self._coef2 = None

    @property
    def nlay(self):
        """
        Get number of layers in the model
        """
        if self._dis is not None:
            return self._dis._nlay
        return

    @property
    def nrow(self):
        """
        Get number of rows in the model
        """
        if self._dis is not None:
            return self._dis._nrow
        return

    @property
    def ncol(self):
        """
        Get number of columns in the model
        """
        if self._dis is not None:
            return self._dis._ncol
        return

    @property
    def ncpl(self):
        """
        Get the number of cells per layer

        """
        if self._dis is not None:
            return self._dis.ncpl
        return

    @property
    def nnodes(self):
        """
        Get the number of cells in the model

        """
        if self._dis is not None:
            return self._dis.nnodes
        return

    @property
    def shape(self):
        """
        Get the shape of the model
        """
        return self.nlay, self.nrow, self.ncol

    @property
    def hold(self):
        """
        Get the head from iteration - 1

        Returns
        -------
            np.ndarray
        """
        if self._hold is None:
            self._hold = self._ic.strt
        return self._hold

    def lrc_to_node(self, lrcs):
        """
        Conversion method of l,r,c data to node number

        Parameters
        ----------
        lrcs : list or np.ndarray

        Returns
        -------
            numpy array, nodes
        """
        multi_index = tuple(np.array(lrcs).T)
        shape = self.shape
        return np.ravel_multi_index(multi_index, shape)

    def add_package(self, package):
        """
        Method to add packages to the GroundwaterFlow model

        Parameters
        ----------
        package : package.Pakbase
            package or stress package object

        """
        from gwflow.packages.stress_package import StressPakBase

        if package.package_type() == "DIS":
            if self._dis is not None:
                raise AssertionError()
            self._dis = package
        elif package.package_type() == "HYD":
            if self._hyd is not None:
                raise AssertionError()
            self._hyd = package
        elif package.package_type() == "STO":
            if self._sto is not None:
                raise AssertionError()
            self._sto = package
        elif package.package_type() == "IC":
            self._ic = package
        else:
            if isinstance(package, StressPakBase):
                if package.package_type() == "CHD":
                    self._chd_pkgs.append(package)
                else:
                    self._stress_pkgs.append(package)

    def calculate_conductance(self):
        """
        Method to calculate conductance for the GroundwaterFlow model

        Index positions in the array are as follows and correspond to the neighbor array:
        0 : column up
        1 : row left
        2 : row right
        3 : column down
        """
        if self._dis is None:
            raise AssertionError()
        if self._hyd is None:
            raise AssertionError()

        if self._dis.package_type() != "DIS":
            raise AssertionError()
        if self._hyd.package_type() != "HYD":
            raise AssertionError()

        neighbors = self._dis.neighbors
        n2n_dist = self._dis.horizontal_node_to_node_distance
        hvwa = self._dis.hvwa
        thick = self._dis.cell_thick
        hk = self._hyd.hk

        conductance = np.full(neighbors.shape, 0., dtype=float)
        for node, neigh in enumerate(neighbors):
            dists = n2n_dist[node]
            wnms = hvwa[node]
            hkn = hk[node]
            vn = thick[node]
            for ix, nn in enumerate(neigh):
                if nn == -1:
                    continue
                hkm = hk[nn]
                vm = thick[nn]
                wnm = wnms[ix]
                cl12 = dists[ix]
                num = ((hkn + hkm) / 2) * wnm * ((vn + vm) / 2)
                cond = num / cl12
                conductance[node, ix] = cond

        return conductance

    def calculate_vertical_conductance(self):
        """
        Method to calculate the vertical conductance for the GroundwaterFlow model

        Returns
        -------
        np.ndarray (nnodes, 2)

        Index positions in the array are as follows and correspond to the neighbor array:
        0 : layer up
        1 : layer down

        """
        if self._dis is None:
            raise AssertionError()
        if self._hyd is None:
            raise AssertionError()

        if self._dis.package_type() != "DIS":
            raise AssertionError()
        if self._hyd.package_type() != "HYD":
            raise AssertionError()

        neighbors = self._dis.vertical_neighbors
        n2n_dist = self._dis.vertical_node_to_node_distance
        area = self._dis.cell_area
        vk = self._hyd.vk

        conductance = np.full(neighbors.shape, 0., dtype=float)
        for node, neigh in enumerate(neighbors):
            ixnode = node
            while ixnode >= self.ncpl:
                ixnode -= self.ncpl

            dists = n2n_dist[node]
            xca = area[ixnode]
            vkn = vk[node]
            for ix, nn in enumerate(neigh):
                if nn == -1:
                    continue
                cl12 = dists[ix]
                vkm = vk[nn]
                num = ((vkn + vkm) / 2) * xca
                cond = num / cl12
                conductance[node, ix] = cond

        return conductance

    def rhs(self, update=True):
        """
        Method to create the RHS vector for solving the GWF equation

        Parameters
        ----------
        update : bool

        Returns
        -------
            np.array
        """
        if update or self._rhs is None:
            rhs = np.full((self.nnodes,), 0., dtype=float)
            for pkg in self._stress_pkgs:
                nodes = pkg.nodes
                bc_rhs = pkg.rhs
                rhs[nodes] += bc_rhs

            for chd in self._chd_pkgs:
                nodes = chd.nodes
                chd_rhs = chd.rhs
                rhs[nodes] = chd_rhs

            self._rhs = rhs

        return self._rhs

    def hcof(self, update=True):
        """
        Head coeficient from boundary condition packages. More or less the package
        conductance values

        Parameters
        ----------
        update : bool

        Returns
        -------
            np.array
        """
        if update or self._hcof is None:
            hcof = np.full((self.nnodes,), 0., dtype=float)
            for pkg in self._stress_pkgs:
                nodes = pkg.nodes
                bc_hcof = pkg.hcof
                hcof[nodes] += bc_hcof

            self._hcof = hcof

        return self._hcof

    def update_coef2(self):
        return 0

    def Amatix(self, update=True):
        """
        Method to create the sparse "A" matrix for solving the GWF equation

        Parameters
        ----------
        update: bool
            method to recalculate the A matrix

        Returns
        -------
            scipy.sparse.diag matrix
        """
        if update or self._amat is None:
            conductance = self.calculate_conductance()
            vcond = self.calculate_vertical_conductance()
            hcof = self.hcof()
            coef2 = self.update_coef2()

            """
            offsets = [self.ncpl, self.ncol, 1, 0, -1, -self.ncol, -self.ncpl]
            mat_coefs = np.zeros((self.nnodes, 7))
            mat_coefs[:, 0] = vcond[:, 0] # layer up
            mat_coefs[:, 1] = conductance[:, 0]  # row up
            mat_coefs[:, 2] = conductance[:, 1]  # left
            mat_coefs[:, 4] = conductance[:, 2]  # right
            mat_coefs[:, 5] = conductance[:, 3]  # down
            mat_coefs[:, 6] = vcond[:, 1] # row down
            mat_coefs = mat_coefs.T
            """
            # EQ. 2-21 in MODFLOW-6 docs
            diag_coef = -1 * (np.sum(conductance, axis=1).ravel() + np.sum(vcond, axis=1).ravel()) + hcof - coef2
            # mat_coefs[3] = diag_coef

            # Now add in constant head package correction if they exist
            # 1 on the diagonal, 0 on the cross term
            # todo: need to rework this....
            #  adjust the diagonal coefficient and then set the conductances to 0
            if self._chd_pkgs:
                for chd in self._chd_pkgs:
                    nodes = chd.nodes
                    for node in nodes:
                        diag_coef[node] = 1
                        conductance[node, :] = 0
                        vcond[node, :] = 0
                    # for idx in range(7):
                    #     coef = 0
                    #     if idx == 3:
                    #         coef = 1
                    #     idx = np.full(nodes.size, idx, dtype=int)
                    #     mat_coefs[idx, nodes] = coef

            # todo: need to create r,c indicies for the matrix coeficients from neighbors
            row = list(range(0, self.nnodes))
            col = list(range(0, self.nnodes))
            data = [i for i in diag_coef]
            for node, neighs in enumerate(self._dis.neighbors):
                for ix, n in enumerate(neighs):
                    if n == -1:
                        continue
                    row.append(node)
                    col.append(n)
                    data.append(conductance[node, ix])

            for node, neighs in enumerate(self._dis.vertical_neighbors):
                for ix, n in enumerate(neighs):
                    if n == -1:
                        continue
                    row.append(node)
                    col.append(n)
                    data.append(vcond[node, ix])


            csr_mat = sp.sparse.csr_array(
                (data, (row, col)),
                shape=(self.nnodes, self.nnodes)
            )
            # check = csr_mat.toarray()

            # amat = sp.sparse.dia_array(
            #      (mat_coefs, offsets),
            #      shape=(self.nnodes, self.nnodes)
            # )
            self._amat = csr_mat
            # x = amat.toarray()
        return self._amat

    def solve(self, maxiters, htol=0.1):
        """
        Simple development solver. Non-transient

        Returns
        -------

        """
        hguess = self.hold
        htol = np.full((self.nnodes,), htol, dtype=float)

        for _ in range(maxiters):
            A = self.Amatix(update=True)
            rhs = self.rhs(update=True)
            h, info = gmres(A, rhs, x0=hguess, rtol=0.00001, atol=0.01)

            if np.abs(np.sum(h - hguess)) > np.abs(np.sum(htol)):
               hguess = h
            else:
                break

        return h
