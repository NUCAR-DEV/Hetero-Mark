from scipy import sparse
import numpy as np
from scipy.sparse import csr_matrix

def genCsrFile(numRows, numCols, density):
    m = sparse.rand(numRows, numCols, density=density, format='csr', dtype=np.float32, random_state=None)
    a = m.toarray()
    zeroCol = np.where(~a.any(axis=0))[0]
    a[0,zeroCol] = 1
    stoMatrix = np.divide(a, np.sum(a, axis=0))
    stoMatrix = csr_matrix(stoMatrix)

#    fileName = './csrMatrix/' + 'csr_'+str(numRows)+'_'+str(int(density*100))+'.txt'
    fileName = 'csr_'+str(numRows)+'_'+str(int(density*100))+'.txt'
    f = open(fileName, 'w')
    print >> f, "{:d}".format(stoMatrix.nnz), "{:d}".format(stoMatrix.shape[0])
    for number in stoMatrix.indptr.tolist():
        print >> f, "{:d}".format(number),
    print >> f, '\n',
    for number in stoMatrix.indices.tolist():
        print >> f, "{:d}".format(number),
    print >> f, '\n',
    for number in stoMatrix.data.tolist():
        print >> f, "{:.2f}".format(number),
    print >> f, '\n',
    f.close()
for numRows in range(1024,10240,1024):
    for density in [0.1]:
        genCsrFile(numRows, numRows, density)
