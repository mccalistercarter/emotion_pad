import numpy as np

def fuse_pad(pads):
    # Filter out missing predictions
    valid = [p for p in pads if p is not None]

    if not valid:
        return (0, 0, 0)
    
    # Have to flatten each vector
    valid = [np.array(p).flatten() for p in valid]

    result = np.mean(valid, axis=0)

    # Return plain float rounded off to two decimal places
    return tuple(round(float(x), 2) for x in result)
