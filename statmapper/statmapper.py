import numpy as np
from scipy.stats import ks_2samp
from scipy.sparse.csgraph import dijkstra, shortest_path, connected_components
from scipy.sparse import csr_matrix
from sklearn_tda import MapperComplex
import networkx as nx
from networkx import cycle_basis
import gudhi as gd
import struct
import matplotlib
import matplotlib.pyplot as plt
import colorsys
import os
import sys

def compute_topological_features(M, func=None, func_type="data", topo_type="downbranch", threshold=0.):
	"""
	Compute the different topological structures associated to a Mapper graph (connected components, downward branches, upward branches, loops).

	Parameters:
		M (mapper graph): Mapper (as computed by sklearn_tda).
		func (list): function used to compute the structures. It is either defined on the Mapper nodes (if func_type = "node") or on the input data (if func_type = "data"). If None, the function is computed with eccentricity.
		func_type (string): type of function used to compute the structures. Either "node" or "data".
		topo_type (string): type of topological structures. Either "connected_components", "downbranch", "upbranch" or "loop".
		threshold (float): threshold on the topological structure types. Structures of size less than this value are ignored.

	Returns:
		dgm (list of tuple (dimension, (vb, vd))): list containing the dimension and the coordinates of each topological structure.
		bnd (list of list of int): data points corresponding to each topological structure.
	"""
	mapper = M.mapper_
	node_info = M.node_info_
	num_pts_mapper = len(node_info)

	if func is None:
		func_type = "graph"
		# Compute inverse of eccentricity
		A = np.zeros([num_pts_mapper, num_pts_mapper])
		for (splx,_) in mapper.get_skeleton(1):
			if len(splx) == 2:	
				A[splx[0], splx[1]] = 1
				A[splx[1], splx[0]] = 1
		dij = dijkstra(A, directed=False)
		D = np.where(np.isinf(dij), np.zeros(dij.shape), dij)
		func = list(-D.max(axis=1))

	function = [np.mean([func[i] for i in node_info[v]["indices"]]) for v in range(num_pts_mapper)] if func_type == "data" else func
	dgm, bnd = [], []

	if topo_type == "connected_components":
		
		num_pts = len(function)
		A = np.zeros([num_pts, num_pts])
		for (splx,_) in mapper.get_skeleton(1):
			if len(splx) == 2:	
				A[splx[0], splx[1]] = 1
				A[splx[1], splx[0]] = 1

		_, ccs = connected_components(A, directed=False)
		
		for ccID in np.unique(ccs):
			pts = np.argwhere(ccs == ccID).flatten()
			vals = [function[p] for p in pts]
			if np.abs(min(vals) - max(vals)) >= threshold:
				dgm.append((0, (min(vals), max(vals))))
				bnd.append(pts)

	if topo_type == "downbranch" or topo_type == "upbranch":

		if topo_type == "upbranch":	function = [-f for f in function]

		def find(i, parents):
			if parents[i] == i:	return i
			else:	return find(parents[i], parents)


		def union(i, j, parents, f):
			if f[i] <= f[j]:	parents[j] = i
			else:	parents[i] = j

		num_pts = len(function)
		A = np.zeros([num_pts, num_pts])
		for (splx,_) in mapper.get_skeleton(1):
			if len(splx) == 2:	
				A[splx[0], splx[1]] = 1
				A[splx[1], splx[0]] = 1

		sorted_idxs = np.argsort(np.array(function))
		inv_sorted_idxs = np.arange(num_pts)
		for i in range(num_pts):	inv_sorted_idxs[sorted_idxs[i]] = i

		diag, comp, parents, visited = {}, {}, -np.ones(num_pts, dtype=np.int32), {}
		for i in range(num_pts):

			current_pt = sorted_idxs[i]
			neighbors = np.argwhere(A[current_pt,:] == 1.)
			if len(neighbors) == 1:	neighbors = neighbors[0,:]
			else:	neighbors = np.squeeze(neighbors) 
			lower_neighbors = [n for n in neighbors if inv_sorted_idxs[n] <= i] if neighbors.shape[0] > 0 else []
			if lower_neighbors == []:
	
				parents[current_pt] = current_pt

			else:

				neigh_pars = [find(n, parents) for n in lower_neighbors]
				g = neigh_pars[np.argmin([function[n] for n in neigh_pars])]
				pg = find(g, parents)
				parents[current_pt] = pg
				for neighbor in lower_neighbors:
					pn = find(neighbor, parents)
					val = max(function[pg], function[pn])
					if pg != pn:
						pp = pg if function[pg] > function[pn] else pn						
						comp[pp] = []
						for v in np.arange(num_pts)[sorted_idxs[:i]]:
							if find(v, parents) == pp:
								try:	visited[v]
								except KeyError:
									visited[v] = True
									comp[pp].append(v)
						comp[pp].append(current_pt)
						if np.abs(function[pp]-function[current_pt]) >= threshold:	diag[pp] = current_pt
						union(pg, pn, parents, function)
					else:						
						if len(neighbors) == len(lower_neighbors):
							comp[pg] = []
							for v in np.arange(num_pts)[sorted_idxs[:i+1]]:
								if find(v, parents) == pg:
									try:	visited[v]
									except KeyError:
										visited[v] = True
										comp[pg].append(v)
							comp[pg].append(current_pt)
							if np.abs(function[pg]-function[current_pt]) >= threshold:	diag[pg] = current_pt
		
		for key, val in iter(diag.items()):
			if topo_type == "downbranch":	dgm.append((0, (function[key],  function[val])))
			elif topo_type == "upbranch":	dgm.append((0, (-function[val], -function[key])))
			bnd.append(comp[key])

	elif topo_type == "loop":

		G = mapper2networkx(M)
		bndall = cycle_basis(G)
		for pts in bndall:
			vals = [function[p] for p in pts]
			if np.abs(min(vals) - max(vals)) >= threshold:	
				dgm.append((1,(min(vals), max(vals))))
				bnd.append(pts)
		
	return dgm, bnd

def evaluate_significance(dgm, bnd, X, M, func, params, topo_type="loop", threshold=.9, N=1000, input="point cloud"):
	"""
	Evaluate the statistical significance of each topological structure of a Mapper graph with bootstrap.

	Parameters:
		dgm (list of tuple (dimension, (vb, vd))): list containing the dimension and the coordinates of each topological structure.
		bnd (list of list of int): data points corresponding to each topological structure.
		X (numpy array of shape n x d if point cloud and n x n if distance matrix): input point cloud or distance matrix.
		M (mapper graph): Mapper (as computed by sklearn_tda).
		func (list): function used to compute the structures. It is either defined on the Mapper nodes (if func_type = "node") or on the input data (if func_type = "data"). If None, the function is computed with eccentricity.
		params (dictionary): parameters used to compute the original Mapper.
		topo_type (string): type of topological structure. Either "connected_components", "downbranch", "upbranch" or "loop".
		threshold (float): threshold on the statistical significance.
		N (int): number of bootstrap iterations.
		input (string): type of input data. Either "point cloud" or "distance matrix".

	Returns:
		dgmboot (list of tuple (dimension, (vb, vd))): subset of dgm with statistical significance above threshold.
		bndboot (list of list of int): subset of bnd with statistical significance above threshold.
	"""
	num_pts, distribution = len(X), []

	for bootstrap_id in range(N):

		# Randomly select points
		idxs = np.random.choice(num_pts, size=num_pts, replace=True)
		Xboot = X[idxs,:] if input == "point cloud" else X[idxs,:][:,idxs]
		f_boot = [func[i] for i in idxs]
		params_boot = {k: params[k] for k in params.keys()}
		params_boot["filters"] = params["filters"][idxs,:]
		params_boot["colors"] = params["colors"][idxs,:]
		Mboot = MapperComplex(**params_boot).fit(Xboot)
		
		# Compute the corresponding persistence diagrams
		dgm_boot, _ = compute_topological_features(Mboot, func=f_boot, func_type="data", topo_type=topo_type)

		# Compute the bottleneck distances between them and keep the maximum
		npts, npts_boot = len(dgm), len(dgm_boot)
		D1 = np.array([[dgm[pt][1][0], dgm[pt][1][1]] for pt in range(npts) if dgm[pt][0] <= 1]) 
		D2 = np.array([[dgm_boot[pt][1][0], dgm_boot[pt][1][1]] for pt in range(npts_boot) if dgm_boot[pt][0] <= 1])
		bottle = gd.bottleneck_distance(D1, D2)
		distribution.append(bottle)

	distribution = np.sort(distribution)
	dist_thresh  = distribution[int(threshold*len(distribution))]
	significant_idxs = [i for i in range(len(dgm)) if dgm[i][1][1]-dgm[i][1][0] >= 2*dist_thresh]	
	dgmboot, bndboot = [dgm[i] for i in significant_idxs], [bnd[i] for i in significant_idxs] 
	return dgmboot, bndboot

def mapper2networkx(mapper, get_attrs=False):
	"""
	Turn a Mapper graph (as computed by sklearn_tda) into a networkx graph.

	Parameters:
		mapper (mapper graph): Mapper (as computed by sklearn_tda).
		get_attrs (bool): whether to use Mapper attributes or not.

	Returns:
		G (networkx graph): networkx graph associated to the Mapper.
	"""
	M = mapper.mapper_
	G = nx.Graph()
	for (splx,_) in M.get_skeleton(1):	
		if len(splx) == 1:	G.add_node(splx[0])
		if len(splx) == 2:	G.add_edge(splx[0], splx[1])
	if get_attrs:
		attrs = {k: {"attr_name": mapper.node_info_[k]["colors"]} for k in G.nodes()}
		nx.set_node_attributes(G, attrs)
	return G


def print_to_dot(M, color_name="viridis", name_mapper="mapper", name_color="color", epsv=.2, epss=.4):
	"""
	Produce a pdf file with a drawing of the Mapper.

	Parameters:
		M (mapper graph): Mapper (as computed by sklearn_tda).
		color_name (string): color map to use for the Mapper nodes.
		name_mapper (string): name of the pdf file.
		name_color (string): name of the color used to color the Mapper nodes.
		epsv (float): minimum of the color map.
		epss (float): maximum of the color map.
	"""
	mapper = M.mapper_
	node_info = M.node_info_

	threshold = 0.
	maxv, minv = max([node_info[k]["colors"][0] for k in node_info.keys()]), min([node_info[k]["colors"][0] for k in node_info.keys()])
	maxs, mins = max([node_info[k]["size"]      for k in node_info.keys()]), min([node_info[k]["size"]      for k in node_info.keys()])  

	f = open(name_mapper + "_" + name_color + ".dot", "w")
	f.write("graph MAP{")
	cols = []
	for (simplex,_) in mapper.get_skeleton(0):
		cnode = (1.-2*epsv) * (node_info[simplex[0]]["colors"][0] - minv)/(maxv-minv) + epsv if maxv != minv else 0
		snode = (1.-2*epss) * (node_info[simplex[0]]["size"]-mins)/(maxs-mins) + epss if maxs != mins else 1
		f.write(  str(simplex[0]) + "[shape=circle width=" + str(snode) + " fontcolor=black color=black label=\""  + "\" style=filled fillcolor=\"" + str(cnode) + ", 1, 1\"]")
		cols.append(cnode)
	for (simplex,_) in mapper.get_filtration():
		if len(simplex) == 2:
			f.write("  " + str(simplex[0]) + " -- " + str(simplex[1]) + " [weight=15];")
	f.write("}")
	f.close()

	L = np.linspace(epsv, 1.-epsv, 100)
	colsrgb = []
	for c in L:	colsrgb.append(colorsys.hsv_to_rgb(c,1,1))
	fig, ax = plt.subplots(figsize=(6, 1))
	fig.subplots_adjust(bottom=0.5)
	my_cmap = matplotlib.colors.ListedColormap(colsrgb, name=color_name)
	cb = matplotlib.colorbar.ColorbarBase(ax, cmap=my_cmap, norm=matplotlib.colors.Normalize(vmin=minv, vmax=maxv), orientation="horizontal")
	cb.set_label(name_color)
	fig.savefig("cbar_" + name_color + ".pdf", format="pdf")
	plt.close()

def compute_DE_features(X, M, nodes, features=None, sparse=False):
	"""
	Compute the differentially expressed features corresponding to some topological structures of a Mapper graph.

	Parameters:
		X (numpy array of shape n x d): input point cloud.
		M (mapper graph): Mapper (as computed by sklearn_tda).
		nodes (list of string): nodes of the Mapper belonging to the topological structure on which DE features are to be computed.
		features (list of int): indices of the features to test for differential accessibility.
		sparse (bool): whether to use sparse representation of the point cloud or not.

	Returns:
		F (array of int): indices of the DE features.
		P (array of float): p-values corresponding to the DE features.
	"""
	node_info = M.node_info_
	
	if features is None:	features = np.arange(X.shape[1])

	list_idxs1 = list(np.unique(np.concatenate([node_info[node_name]["indices"] for node_name in nodes])))
	list_idxs2 = list(set(np.arange(X.shape[0]))-set(list_idxs1))
	pvals = []
	for f in features:
		if sparse:
			Xsp = csr_matrix(X)
			group1, group2 = np.squeeze(np.array(Xsp[list_idxs1,f].todense())), np.squeeze(np.array(Xsp[list_idxs2,f].todense()))
		else:
			group1, group2 = X[list_idxs1,f], X[list_idxs2,f]
		_,pval = ks_2samp(group1, group2)
		pvals.append(pval)
	pvals = np.array(pvals)
	F, P = features[np.argsort(pvals)], np.sort(pvals) 
	return F, P
	
