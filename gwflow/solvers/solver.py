import numpy as np


class Solver:
    """
    Base solver class, not to be instantiated directly by user

    Parameters
    ----------
    model : GroundwaterFlow object
    mxiters : int
        maximum number of iterations
    rclose : float
        residual closure criteria (cell x cell basis)
    """
    def __init__(self, model, mxiter, rclose, outer_close):
        self._model = model
        self._mxiter = mxiter
        self._rclose = rclose
        self._outer_close = outer_close

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
        while niter < self._mxiter:
            Amat = self._model.Amatix(update=True)
            rhs = self._model.rhs(update=True)
            hold = self._model.hold

            hd = self.inner_solve(Amat, rhs, hold)

            # if (Ax - rhs) > resid of rhs, iterate again and recalculate Amat and rhs using new heads
            est_b = Amat.dot(hd)
            resid = np.abs(est_b - rhs)

            ixs = np.where(resid > self._outer_close)[0]
            if len(ixs) > 0:
                # todo: need to pass new hold through HCOF and RHS calculations
                # hold = hd.copy()
                # todo: hack for moment is setting hold to self._model._hold
                self._model._hold = hd.copy()
                niter += 1
                if niter == self._mxiter:
                    raise Exception(f"Outer solution did not converge after {self._mxiter} iterations")
            else:
                # converged
                niter += 1
                break

        return hd

    def solve(self):
        raise AssertionError("must be specified in child class")