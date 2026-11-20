"""Recoverable detector exceptions used by the archived clinical inference routine."""
import numpy as np


def recoverable(exc):
    msg = str(exc)
    return (("out of bound" in msg and "FecgQRSmCanc" in msg)
            or "Too few preliminary Power-MF peaks" in msg
            or "Expected positive acquisition frequency and at least five seconds" in msg
            or isinstance(exc, np.linalg.LinAlgError))
