import numpy as np
from scipy.stats import ks_2samp
from scipy.sparse.csgraph import dijkstra
from scipy.sparse import csr_matrix
from sklearn_tda import MapperComplex
import networkx as nx
import gudhi as gd
import struct
import matplotlib
import matplotlib.pyplot as plt
import colorsys
import os
import sys

def write_input_HomLoc(name, sc, vals):

	dimension, num_pts = sc.dimension(), len(sc.get_skeleton(0))
	pos = [0 for _ in range(num_pts)]

	fout = open(name + ".dat", "wb")
	fout.write(struct.pack("i", 2))
	fout.write(struct.pack("i", dimension))
	fout.write(struct.pack("i", num_pts))
	fout.write(struct.pack("i", 1))
	fout.write(struct.pack("d" * num_pts, *pos))
	fout.write(struct.pack("d" * num_pts, *vals))
	for dim in range(dimension+1):
		list_splxs = [s for (s,_) in sc.get_skeleton(dim) if len(s) == dim+1]
		fout.write(struct.pack("ii", dim, len(list_splxs)))
		for spl in list_splxs:	
			fout.write(struct.pack("i" * (dim+1), *spl))
	fout.close()


def read_output_HomLoc(name, homology):
	
	dgm = []
	fdgm = open(name + ".dat.pers.txt", "r")
	L = fdgm.readlines()

	for line in L:
		
		if line == "\n":	continue
		else: 
			if line[0] == "[":	
				homdim, num_pts = int(line[1]), int(line[-2])
				continue
			else:
				lline = line[:-1].split("\t")
				dgm.append((homdim, tuple([float(num) for num in lline])))
		
	bd, rd = [], []
	fbnd = open(name + ".dat.bnd." + str(homdim+1), "r")
	bnd = np.fromfile(fbnd, np.int16)
	bnd = bnd[bnd>0].astype(np.int32)

	if len(bnd) == 1:
		bd.append([])
	else:
		ndim, ncyc = bnd[0], bnd[1]
		b, cursor = [], 2
		for _ in range(ncyc):
			pts_in_cyc = bnd[cursor]
			b.append(np.squeeze(np.reshape(bnd[cursor+1:cursor+pts_in_cyc*ndim+1], [pts_in_cyc, ndim])-1))
			cursor += pts_in_cyc*ndim+1
		bd.append(b)

	fred = open(name + ".dat.red." + str(homdim+1), "r")
	red = np.fromfile(fred, np.int16)
	red = red[red>0].astype(np.int32)

	if len(red) == 1:
		rd.append([])
	else:
		ndim, ncyc = red[0], red[1]
		r, cursor = [], 2
		for _ in range(ncyc):
			pts_in_cyc = red[cursor]
			r.append(np.squeeze(np.reshape(red[cursor+1:cursor+pts_in_cyc*ndim+1], [pts_in_cyc, ndim])-1))
			cursor += pts_in_cyc*ndim+1
		rd.append(r)

	return dgm, bd, rd






def compute_topological_features(M, func, topo_type="loop", path_to_homology_localization="./HomologyLocalization"):

	mapper = M.mapper_
	node_info = M.node_info_
	num_pts_mapper = len(node_info)
	function = [np.mean([func[i] for i in node_info[v]["indices"]]) for v in range(num_pts_mapper)]

	if topo_type == "downbranch" or topo_type == "upbranch":

		if topo_type == "upbranch":	function = [-f for f in function]

		def find(i, parents):
			if parents[i] == i:	return i
			else:	return find(parents[i], parents)


		def union(i, j, parents, f):
			if f[i] < f[j]:	parents[j] = i
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

		diag, comp, parents = {}, {}, -np.ones(num_pts, dtype=np.int32)
		for i in range(num_pts):

			current_pt = sorted_idxs[i]
			neighbors = np.squeeze(np.argwhere(A[current_pt,:] == 1.))
			lower_neighbors = [n for n in neighbors if inv_sorted_idxs[n] <= i] if len(neighbors.shape) > 0 else []

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
						comp[pp] = [v for v in np.arange(num_pts)[sorted_idxs[:i]] if find(v, parents) == pp]
						diag[pp] = current_pt
						union(pg, pn, parents, function)
		dgm, bnd = [], []
		for key, val in iter(diag.items()):
			if topo_type == "downbranch":	dgm.append((0, (function[key],  function[val])))
			elif topo_type == "upbranch":	dgm.append((0, (-function[val], -function[key])))
			pts = comp[key]			
			minI, maxI = np.argmin([function[p] for p in pts]), np.argmax([function[p] for p in pts])
			subA = A[comp[key],:][:,comp[key]]
			dists, preds = dijkstra(subA, return_predecessors=True)
			init_pt = maxI
			final_pts = [pts[init_pt]]
			while init_pt is not minI:
				init_pt = preds[minI, init_pt]
				if init_pt == -9999:	break
				final_pts.append(pts[init_pt])
			bnd.append(final_pts)

	elif topo_type == "loop":

		sc = gd.SimplexTree()
		maxf, num_pts = max(function), len(mapper.get_skeleton(0))
		for (splx,f) in mapper.get_skeleton(1):	
			sc.insert(splx)
			sc.insert(splx + [num_pts])
		func = function + [maxf+1.]

		write_input_HomLoc("tmp", sc, func)
		os.system(path_to_homology_localization + " -f tmp.dat")
		_, bound, _ = read_output_HomLoc("tmp", 1)

		dgm, bnd = [], []
		for cycle in bound:
			if cycle == []:	continue
			cycle = np.squeeze(cycle)
			bnd.append(cycle)
			dgm.append((1,(min([function[v] for v in cycle]), max([function[v] for v in cycle]))))
		
	return dgm, bnd

def evaluate_significance(dgm, bnd, X, M, func, params, topo_type="loop", threshold=.9, N=1000, input="point cloud", path_to_homology_localization="./HomologyLocalization"):
	
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
		dgm_boot, _ = compute_topological_features(Mboot, f_boot, topo_type=topo_type, path_to_homology_localization=path_to_homology_localization)

		# Compute the bottleneck distances between them and keep the maximum
		npts, npts_boot = len(dgm), len(dgm_boot)
		D1 = np.array([[dgm[pt][1][0], dgm[pt][1][1]] for pt in range(npts) if dgm[pt][0] <= 1]) 
		D2 = np.array([[dgm_boot[pt][1][0], dgm_boot[pt][1][1]] for pt in range(npts_boot) if dgm_boot[pt][0] <= 1])
		bottle = gd.bottleneck_distance(D1, D2)
		distribution.append(bottle)

	distribution = np.sort(distribution)
	dist_thresh  = distribution[int(threshold*len(distribution))]
	significant_idxs = [i for i in range(len(dgm)) if dgm[i][1][1]-dgm[i][1][0] >= 2*dist_thresh]	

	return [dgm[i] for i in significant_idxs], [bnd[i] for i in significant_idxs]

def mapper2networkx(mapper):
	M = mapper.mapper_
	G = nx.Graph()
	for (splx,_) in M.get_skeleton(1):	
		if len(splx) == 1:	G.add_node(splx[0])
		if len(splx) == 2:	G.add_edge(splx[0], splx[1])
	attrs = {k: {str(c): mapper.node_info_[k]["colors"][c] for c in range(len(mapper.node_info_[k]["colors"]))} for k in G.nodes()}
	nx.set_node_attributes(G, attrs)
	return G

def compute_average(mapper_list, N=None, path_to_fgw="/home/mathieu/Documents/code/fgw/lib"):

	if N is None:	N = int(np.mean(np.array([len(mapper.mapper_.get_skeleton(0)) for mapper in mapper_list])))

	sys.path.append(path_to_fgw)
	from FGW import fgw_barycenters
	from graph import Graph, graph_colors, find_thresh, sp_to_adjency

	graph_list    = [Graph(mapper2networkx(mapper)) for mapper in mapper_list]
	Cs            = [x.distance_matrix(force_recompute=True, method="shortest_path") for x in graph_list]
	ps            = [np.ones(len(x.nodes()))/len(x.nodes()) for x in graph_list]
	Ys            = [[v for (k,v) in nx.get_node_attributes(x.nx_graph, "0").items()] for x in graph_list] #[x.values() for x in graph_list]
	lambdas       = np.array([np.ones(len(Ys))/len(Ys)]).ravel()
	init_X        = np.repeat(N,N)
	D1, C1, log   = fgw_barycenters(N, Ys, Cs, ps, lambdas, alpha=0.95, init_X=init_X)
	bary          = nx.from_numpy_matrix(sp_to_adjency(C1, threshinf=0, threshsup=find_thresh(C1, sup=100, step=100)[0]))
	for i in range(len(D1)):	bary.add_node(i, attr_name=float(D1[i]))
	nc = graph_colors(bary, vmin=-1, vmax=1)
	return bary, nc

def print_to_dot(M, color_name="viridis", name_mapper="mapper", name_color="color", epsv=.2, epss=.4):

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

def compute_DE_features(X, M, nodes, out_feats=10, features=None, sparse=False):

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
	
	return features[np.argsort(pvals)[:out_feats]], np.sort(pvals)
	
