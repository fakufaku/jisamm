import numpy as np


# Compute the demixed output
def demix(Y, X, W):
    Y[:, :, :] = np.matmul(W, X)


def tensor_H(T):
    return np.conj(T).swapaxes(-2, -1)


class TwoStepsIterator(object):
    """
    Iterates two elements at a time between 0 and m - 1
    """

    def __init__(self, m):
        self.m = m

    def _inc(self):
        self.next = (self.next + 1) % self.m
        self.count += 1

    def __iter__(self):
        self.count = 0
        self.next = 0
        return self

    def __next__(self):

        if self.count < 2 * self.m:

            m = self.next
            self._inc()
            n = self.next
            self._inc()

            return m, n

        else:
            raise StopIteration


