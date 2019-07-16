from statmapper import compute_topological_features
import gudhi as gd
import os

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

function = [0.,1.,2.,3.,4.,5.,2.,3.,4.,0.5,4.,5.]

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

dgm, bnd = compute_topological_features(mapper, function, "loop", "../../homloc/HomologyLocalization")
print(dgm, bnd)
if len(dgm) > 0:
	plot = gd.plot_persistence_diagram(dgm)
	plot.show()
