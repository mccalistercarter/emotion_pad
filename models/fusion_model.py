import numpy as np

def fuse_pad(pads):
    valid = [p for p in pads if p is not None]
    if not valid:
        return (0.5, 0.5, 0.5)
    result = np.mean(valid, axis=0)
    scaled = 2 * result - 1  # scale from [0, 1] to [-1, 1]

    return tuple(round(float(x), 2) for x in scaled)  # casted to plain floats
