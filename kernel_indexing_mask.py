import numpy as np
from dataclasses import dataclass
from icecream import ic


@dataclass
class GridDim:
    x: float = 1
    y: float = 1
    z: float = 1


@dataclass
class BlockIdx:
    x: float
    y: float
    z: float


@dataclass
class BlockDim:
    x: float = 1
    y: float = 1
    z: float = 1


@dataclass
class ThreadIdx:
    x: float
    y: float
    z: float


if __name__ == "__main__":
    A = np.arange(6 * 12).reshape(6, 12)
    ic(A)

    # Thread block
    blockDim = BlockDim(3, 5, 1)
    gridDim = GridDim(2, 3, 1)

    b = np.zeros((6 * 12,))
    c = np.zeros((6 * 12,))
    d = np.zeros((6 * 12,))
    e = np.zeros((6 * 12,))

    def kernel(blockIdx: BlockIdx, threadIdx: ThreadIdx):
        i = blockIdx.x * blockDim.x + threadIdx.x
        j = blockIdx.y * blockDim.y + threadIdx.y

        idx = j + i * blockDim.y * gridDim.y
        if idx < 6 * 12:
            b[idx] = i
            c[idx] = j

        if i < 6 and j < 12:
            idx = j + i * 12
            d[idx] = i
            e[idx] = j

    for bk in range(gridDim.z):
        for bj in range(gridDim.y):
            for bi in range(gridDim.x):
                blockIdx = BlockIdx(bi, bj, bk)
                for tk in range(blockDim.z):
                    for tj in range(blockDim.y):
                        for ti in range(blockDim.x):
                            threadIdx = ThreadIdx(ti, tj, tk)
                            kernel(blockIdx, threadIdx)

    B = b.reshape(6, 12)
    C = c.reshape(6, 12)
    D = d.reshape(6, 12)
    E = e.reshape(6, 12)

    ic(B)
    ic(C)
    ic(D)
    ic(E)
