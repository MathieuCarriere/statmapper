import numpy as np
from scipy.stats import ks_2samp

def compute_persistence_diagram(mapper, function, show=False):

	for (vl,_) in mapper.get_skeleton(0):
		mapper.assign_filtration(vl, function[vl[0]])
	mapper.initialize_filtration()
	mapper.make_filtration_non_decreasing()
	dgm = mapper.persistence()

	if show:
		plot = gd.plot_persistence_diagram(dgm)
		plot.show()

	return dgm

def compute_DE_features(X, node_info, nodes, out_feats=10, features=None):

	if features is None:	features = np.arange(X.shape[1])
	list_idxs1 = list(np.unique(np.vstack([node_info[node_name]["indices"] for node_name in nodes])))
	list_idxs2 = list(set(np.arange(X.shape[0]))-set(list_idxs1))
	Xsub1, Xsub2 = X[list_idxs1, :], X[list_idxs2, :]

	pvals = []
	for f in features:
		group1, group2 = Xsub1[:,f], Xsub2[:,f]
		_,pval = ks_2samp(group1, group2)
		pvals.append(pval)
	pvals = np.array(pvals)
	
	return features[np.argsort(pvals)[:out_feats]]
	
