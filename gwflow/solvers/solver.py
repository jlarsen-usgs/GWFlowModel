import numpy as np


class Solver:
    """
    Base solver class, not to be instantiated directly by user

    Parameters
    ----------
    model : GroundwaterFlow object
    mxoutiter : int
        maximum number of outer iterations
    mxiniter : int
        maximum number of inner iterations
    hclose : float
        residual closure criteria (cell x cell basis) for the
        inner iteration loop (head based)
    rhs_close : float
        residual closure criteria (cell x cell based) for the
        outer iteration loop (rhs, flux based)
    """
    def __init__(self, model, mxoutiter, mxiniter, hclose, rhs_close):
        self._model = model
        self._mxoutiter = mxoutiter
        self._mxiniter = mxiniter
        self._hclose = hclose
        self._rhs_close = rhs_close

        self._model.add_solver(self)

    def outer_solve(self):
        """
        Generalized method for performing non-linear outer solver iteration

        This method evaluates Ax - b = resid and if resid is greater
        than closure criteria it recalculates A and b (rhs) based
        on newly estimated heads and calls another inner iteration

        Returns
        -------
            hd : np.array
        """
        niter = 0
        hd = self._model.hold.copy()
        while niter < self._mxoutiter:
            Amat = self._model.Amatix(update=True)
            rhs = self._model.rhs(update=True)
            hold = self._model.hold

            hd = self.inner_solve(Amat, rhs, hold)

            # if (Ax - rhs) > resid of rhs, iterate again and recalculate Amat and rhs using new heads
            est_b = Amat.dot(hd)
            resid = np.abs(est_b - rhs)

            ixs = np.where(resid > self._rhs_close)[0]
            if len(ixs) > 0:
                # todo: need to pass new hold through HCOF and RHS calculations
                # todo: hack for moment is setting hold to self._model._hold
                self._model._hold = hd.copy()
                niter += 1
                if niter == self._mxoutiter:
                    raise Exception(f"Outer solution did not converge after {niter} iterations")
            else:
                # converged
                niter += 1
                break

        return hd

    def inner_solve(self, A, b, hold):
        raise AssertionError("must be specified in child class")