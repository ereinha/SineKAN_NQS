def ensure_dir(filename):
    import os

    dirname = os.path.dirname(filename)
    if dirname:
        try:
            os.makedirs(dirname)
        except OSError:
            pass


def init_out_dir():
    from args import args

    if not args.full_out_dir:
        return
    ensure_dir(args.full_out_dir)


def leaf_size_real_nonzero(x):
    from typing import get_args

    import numpy as np
    from netket.utils.types import Array

    if not isinstance(x, get_args(Array)):
        return 0

    # If some but not all elements are exactly float zero, that means they are masked
    size = (x != 0).sum()
    if size == 0:
        size = x.size

    if np.iscomplexobj(x):
        size *= 2

    return size


def tree_size_real_nonzero(tree):
    import jax

    return sum(jax.tree.leaves(jax.tree.map(leaf_size_real_nonzero, tree)))