import cloud
from mpi4py import MPI
import numpy
import sys

comm = MPI.COMM_SELF.Spawn('python',
                           args=['MPISpawn.py'],
                           maxprocs=4)

N = numpy.array(100000, 'i')
comm.Bcast([N, MPI.INT], root=MPI.ROOT)
PI = numpy.array(0.0, 'd')
comm.Reduce(None, [PI, MPI.DOUBLE],
            op=MPI.SUM, root=MPI.ROOT)
print(PI)

comm.Disconnect()
