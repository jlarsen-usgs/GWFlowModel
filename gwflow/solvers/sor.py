import numpy as np
from scipy.sparse import diags
from .solver import Solver


class SorSolver(Solver):


    def __init__(self, model, mxiter=100, rclose=1e-2, outer_close=1e-1, relax=1):

        super().__init__(model, mxiter, rclose, outer_close)

        self._outer_close = outer_close
        if not 0 < relax < 2:
            raise AssertionError(f"{relax=} is out of bounds: (0, 2)")
        self._relax = relax

    def outer_solve(self):
        """
        Entry point for outer solution, defined in Solver
        parent class

        Returns
        -------

        """
        hd = super().outer_solve()
        return hd

    def inner_solve(self, A, b, iguess):
        """

        Parameters
        ----------
        iguess : np.ndarray
            vector of initial condition/guess for the solver

        Returns
        -------

        """
        iter = 0
        x0 = np.array(iguess)
        nx = x0.shape[0]

        resid = A.dot(x0) - b

        diag = A.diagonal()
        xA = A - diags(diag, format="csr")
        indptr = xA.indptr
        indices = xA.indices
        # L = L.toarray()

        # P = ((D / self._relax) + L) * ((self._relax / (2 - self._relax)) * Dinv) * ((D / self._relax) + U)

        while iter < self._mxiter:

            x_old = x0.copy()
            # do stuff
            for i in range(nx):
                LUdata = xA.data[indptr[i]: indptr[i + 1]]
                cols = indices[indptr[i]: indptr[i + 1]]
                # sigma needs the dimension of x0
                sigma = np.dot(LUdata, x0[cols])
                x0[i] = (1 - self._relax) * x0[i] + self._relax * (b[i] - sigma) / diag[i]

            resid = np.abs(x0 - x_old)
            ixs = np.where(resid > self._rclose)[0]
            if len(ixs) > 0:
                x_old = x0.copy()
                iter += 1
                if iter == self._mxiter:
                    raise Exception(f"Inner solution did not converge in {self._mxiter} iterations")
            else:
                iter += 1
                break

        return x0
