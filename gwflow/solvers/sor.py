import numpy as np
from scipy.sparse import diags
from .solver import Solver


class SorSolver(Solver):
    """
    Successive over-relaxation solver

    Parameters
    ----------
    model : gwflow.models.Model
    mxoutiter : int
        maximum number of outer iterations
    mxiniter : int
        maximum number of inner iterations
    hclose : float
        inner head closure tolerance
    outer_close : float
        outer rhs flux closure tolerance
    relax : float
        under/over relaxation factor. 1 is no relaxation,
        0 < relax < 1 is under relaxation (smaller convergence steps), and
        1 < relax < 2 is over relaxation (larger convergence steps)
    """

    def __init__(
        self,
        model,
        mxoutiter=100,
        mxiniter=100,
        hclose=1e-2,
        outer_close=1e-1,
        relax=1
    ):

        super().__init__(model, mxoutiter, mxiniter, hclose, outer_close)

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

    def inner_solve(self, A, b, hold):
        """
        Successive over-relaxation solver for the linear A*x = b
        system

        Parameters
        ----------
        A : scipy.sparse.csr_matrix
            head coeficient matrix
        b : np.ndarray
            rhs flux vector
        hold : np.ndarray
            vector of initial condition/guess for the solver

        Returns
        -------
            hd : np.ndarray, array of heads from the linear solution
        """
        niter = 0
        x0 = np.array(hold)
        nx = x0.shape[0]

        diag = A.diagonal()
        xA = A - diags(diag, format="csr")
        indptr = xA.indptr
        indices = xA.indices
        x_old = x0.copy()

        while niter < self._mxiniter:
            for i in range(nx):
                LUdata = xA.data[indptr[i]: indptr[i + 1]]
                cols = indices[indptr[i]: indptr[i + 1]]
                # sigma needs the dimension of x0
                sigma = np.dot(LUdata, x0[cols])
                x0[i] = (1 - self._relax) * x0[i] + self._relax * (b[i] - sigma) / diag[i]

            resid = np.abs(x0 - x_old)
            ixs = np.where(resid > self._hclose)[0]
            if len(ixs) > 0:
                x_old = x0.copy()
                niter += 1
                if niter == self._mxiniter:
                    raise Exception(f"Inner solution did not converge in {self._mxiniter} iterations")
            else:
                niter += 1
                break

        return x0
