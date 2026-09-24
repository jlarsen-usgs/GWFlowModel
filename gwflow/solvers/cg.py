import numpy as np
from scipy.sparse.linalg import spilu
from .solver import Solver


class ConjugateGradient(Solver):
    """
    Conjugate gradient solver

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
    precondition : bool
        preconditioner flag default is True, when applied
        an approximation of incomplete Cholesky
        preconditioning for the inner solution
        using incomplete ILU preconditioning
    pc_drop_tol : float
        preconditioner drop tolerance (0, 1) default
        is 1e-4
    pc_fill_lev : float
        precondtioner fill level, default is 10
    polak_ribiere_beta : bool
        boolean flag to use Polack-Ribiere Beta calculation
        instead of the Fletcher-Reeves formaulation. Can
        improve convergence for unstable solutions. Default
        is True.
        See https://en.wikipedia.org/wiki/Nonlinear_conjugate_gradient_method
        for more information.

    """
    def __init__(
        self,
        model,
        mxoutiter=100,
        mxiniter=100,
        hclose=1e-4,
        outer_close=1e-5,
        recalc_flux_resid=10,
        precondition=True,
        pc_drop_tol=1e-4,
        pc_fill_lev=10,
        polak_ribiere_beta=True
    ):

        super().__init__(model, mxoutiter, mxiniter, hclose, outer_close)
        self._recalc_flux_resid = recalc_flux_resid
        self._precondition = precondition
        self._drop_tol = pc_drop_tol
        self._fill_lev = pc_fill_lev
        self._pr_beta = polak_ribiere_beta

    def outer_solve(self):
        """

        Returns
        -------

        """
        hd = super().outer_solve()
        return hd

    def preconditioned_inner_solve(self, A, b, hold):
        """
        Preconditioned Conjugate gradient solver for A*x = b
        using incomplete ILU preconditioning

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
        x0 = np.array(hold)
        rk0 = b - A.dot(x0)
        M = spilu(A, drop_tol=self._drop_tol, fill_factor=self._fill_lev)
        zk0 = M.solve(rk0)
        pk0 = zk0.copy()
        niter = 0

        x_old = x0.copy()
        while niter < self._mxiniter:
            Ap = A.dot(pk0)
            alpha = (np.dot(rk0.T, zk0)) / (np.dot(pk0.T, Ap))
            x0 = x_old + alpha * pk0
            if (niter + 1) % self._recalc_flux_resid == 0:
                rk1 = b - A.dot(x0)
            else:
                rk1 = rk0 - Ap * alpha

            zk1 = M.solve(rk1)
            # use the Polak-Ribiere formulation for beta to speed
            # convergence
            if self._pr_beta:
                dzk = zk1 - zk0
                beta = ((np.dot(rk1.T, dzk)) / (np.dot(rk0.T, zk0)))
            else:
                beta = ((np.dot(rk1.T, zk1)) / (np.dot(rk0.T, zk0)))
            pk1 = zk1 + beta * pk0

            resid = np.abs(x0 - x_old)
            rix = np.where(resid > self._hclose)[0]
            if len(rix) > 0:
                pk0 = pk1
                rk0 = rk1
                zk0 = zk1
                x_old = x0.copy()
                niter += 1
                if niter == self._mxiniter:
                    raise Exception(f"Inner solution did not converge in {self._mxiniter} iterations")
            else:
                niter += 1
                break

        return x0

    def inner_solve(self, A, b, hold):
        """
        ConjugateGradient solver for the linear A*x = b
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
        if self._precondition:
            return self.preconditioned_inner_solve(A, b, hold)

        x0 = np.array(hold)
        rk0 = b - A.dot(x0)
        pk0 = rk0.copy()
        niter = 0

        x_old = x0.copy()
        while niter < self._mxiniter:
            Ap = A.dot(pk0)
            alpha = (np.dot(rk0.T, rk0)) / (np.dot(pk0.T, Ap))
            x0 = x_old + alpha * pk0
            if (niter + 1) % self._recalc_flux_resid == 0:
                rk1 = b - A.dot(x0)
            else:
                rk1 = rk0 - Ap * alpha
            beta = -1 * ((np.dot(rk1.T, Ap)) / (np.dot(pk0.T, Ap)))
            pk1 = rk1 + beta * pk0

            resid = np.abs(x0 - x_old)
            rix = np.where(resid > self._hclose)[0]
            if len(rix) > 0:
                pk0 = pk1
                rk0 = rk1
                x_old = x0.copy()
                niter += 1
                if niter == self._mxiniter:
                    raise Exception(f"Inner solution did not converge in {self._mxiniter} iterations")
            else:
                niter += 1
                break

        return x0
