from statmapper import compute_topological_features
import sklearn_tda as sktda
import gudhi as gd
import os
import numpy as np
from scipy.sparse.csgraph import dijkstra

mapper = gd.SimplexTree()
mapper.insert([0,1])
mapper.insert([1,2])
mapper.insert([2,3])
mapper.insert([3,4])
mapper.insert([4,5])
mapper.insert([6,7])
mapper.insert([7,8])
mapper.insert([8,5])
mapper.insert([9,3])
mapper.insert([7,10])
mapper.insert([10,11])

M = sktda.MapperComplex(filters=np.array([[0.]]), filter_bnds=np.array([[np.nan, np.nan]]), colors=np.array([[0.]]), resolutions=np.array([10]), gains=np.array([.3]))
M.mapper_ = mapper
M.node_info_ = {i: [] for i in range(12)}

#	     5   11
#	     |\  |
#	     4 8 10
#	     | |/
#	     3 7
#	    /| |
#	   / 2 6
#	  /  |
#	 /   1
#	9    |
#	     0

dgm, bnd = compute_topological_features(M, topo_type="connected_components")
print(dgm, bnd)
dgm, bnd = compute_topological_features(M, topo_type="downbranch")
print(dgm, bnd)
dgm, bnd = compute_topological_features(M, topo_type="upbranch")
print(dgm, bnd)
dgm, bnd = compute_topological_features(M, topo_type="loop")
print(dgm, bnd)
