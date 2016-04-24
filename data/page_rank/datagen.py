from scipy import sparse
import numpy as np
from scipy.sparse import csr_matrix

def genCsrFile(numRows, numCols, density, filename):
    m = sparse.rand(numRows, numCols, 
            density=density, format='csr', 
            dtype=np.float32, random_state=None)
    a = m.toarray()
    zeroCol = np.where(~a.any(axis=0))[0]
    a[0,zeroCol] = 1
    stoMatrix = np.divide(a, np.sum(a, axis=0))
    stoMatrix = csr_matrix(stoMatrix)

    f = open(filename, 'w')
    print >> f, "{:d}".format(stoMatrix.nnz), "{:d}".format(stoMatrix.shape[0])
    for number in stoMatrix.indptr.tolist():
        print >> f, "{:d}".format(number),
    print >> f, '\n',
    for number in stoMatrix.indices.tolist():
        print >> f, "{:d}".format(number),
    print >> f, '\n',
    for number in stoMatrix.data.tolist():
        print >> f, "{:.20f}".format(number),
    print >> f, '\n',
    f.close()


genCsrFile(64, 64, 0.05, 'tiny.data')
genCsrFile(1024, 1024, 0.1, 'small.data')
genCsrFile(4096, 4096, 0.1, 'medium.data')
genCsrFile(16384, 16384, 0.1, 'large.data')

