#!python2

import pandas as pd
import numpy as np
import msgpack
import msgpack_numpy as m
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import csv
import os

from networkx.drawing.nx_agraph import write_dot

##### Script for analyzing the network of patents (in/out degrees, eigenvector centrality, etc.) and graphing #####

# Patch msgpack_numpy so that msgpack can serialize numpy objects
m.patch()

# Initialize variables
start_year = 1835 # Default: 1835
end_year = 2015 # Default 2015
year_gap = 10
years_to_graph = [1840, 1860, 1880, 1900, 1920, 1940, 1960, 1980, 2000]
network_to_use = 'uspto' # uspto or ipc108 or ipc8
years_per_aggregate = 5 # number of years of data in each matrix/vector

# Loads the vectors and adjacency matrixes
def load_network(network_to_use):
	with open('./cache/' + network_to_use + '/vectors.msgpack', 'rb') as f:
			vectors = msgpack.unpack(f)
	with open('./cache/' + network_to_use + '/matrices.msgpack', 'rb') as f:
			matrices = msgpack.unpack(f)
	with open('./cache/' + network_to_use + '/dictionary.msgpack', 'rb') as f:
			cat_dict = msgpack.unpack(f)
	return vectors, matrices, cat_dict

# Calculate in and out degrees for all the categories in the network over time
def calculate_degrees(adj_matrices, vectors):
	# Initialize the unnormalized and normalized in and out degree vectors
	# Each is a list of vectors for each year, where index of vector is the category index
	# and col1 = out-degree and col2 = in-degree
	num_years = len(adj_matrices)
	n = len(adj_matrices[0])
	unnormalized_degrees = [np.zeros((n, 2)) for i in range(num_years)] # Create an Nx2 vector for each year
	normalized_degrees = [np.zeros((n, 2)) for i in range(num_years)] # Create an Nx2 vector for each year

	# Calculate the in and out degrees for each category for each year
	for year, matrix in enumerate(adj_matrices):
		# Create a networkx multi directed graph
		G = nx.from_numpy_matrix(matrix, create_using=nx.DiGraph())
		# Iterate through each category
		for category in range(len(vectors[year])):
			# Find the degrees for each category
			total_patents = vectors[year][category]
			in_degree = G.in_degree(category, weight='weight')
			out_degree = G.out_degree(category, weight='weight')

			# Find normalized values
			if total_patents == 0:
				normalized_out_degree = 0
				normalized_in_degree = 0
			else: 
				normalized_out_degree = float(out_degree)/total_patents
				normalized_in_degree = float(in_degree)/total_patents

			# Write degree values into the list of vectors
			unnormalized_degrees[year][category] = [out_degree, in_degree]
			normalized_degrees[year][category] = [normalized_out_degree, normalized_in_degree]

	return unnormalized_degrees, normalized_degrees

# Calculate the eigenvector centrality for the network
def calculate_eigenvector_centrality(network_to_use, adj_matrices, years_per_aggregate):
	# List to be populated with centrality measures for each year
	rankings_by_year = []

	# Aggregate the years of the matrices together
	aggregated_matrices = aggregate_years(adj_matrices, years_per_aggregate)

	# Loop through each matrix and calculate the eignvector, then rank each patent by weight
	for i, matrix in enumerate(aggregated_matrices):
		curr_year = start_year + i * years_per_aggregate
		# Convert matrix to a MultiDiGraph
		G = nx.from_numpy_matrix(matrix, create_using=nx.MultiDiGraph())
		# Find eigenvector centrality
		centrality = nx.eigenvector_centrality_numpy(G, weight='weight')
		# Maps matrix index value to a uspto value
		reverse_uspto_dict = {v: k for k, v in cat_dict.iteritems()}
		# Create a list of rankings, where the most central patents are first
		rankings = [[reverse_uspto_dict[k], v] for k, v in centrality.iteritems()]
		rankings = sorted(rankings, key = lambda x: x[1])[::-1] # Sort by value
		# Add 2 entries to the rankings_by_year array (one for the patent name, other for eigenvector)
		rankings_by_year.append((curr_year, [x[0] for x in rankings]))
		rankings_by_year.append((curr_year, [x[1] for x in rankings]))

	# Write into a csv file
	years = [x[0] for x in rankings_by_year] # Extract the years to form the first row
	rankings = [x[1] for x in rankings_by_year] # Extract rankings
	transposed_rankings = zip(*rankings)
	f_name = './cache/' + network_to_use + '/rankings_' + str(years_per_aggregate) + 'year_aggregates.csv'
	with open(f_name, 'wb') as f:
		writer = csv.writer(f)
		writer.writerow(years)
		for row in transposed_rankings:
			writer.writerow(row)

	return rankings_by_year

# Graph the networks for the years of interest
def graph_network(adj_matrices, vectors, start_year, years_of_interest):
	# Load in the crosswalk dictionary for 8 categories
	with open('./cache/ipc8/cw_dictionary.msgpack', 'rb') as f:
		cw_dict = msgpack.unpack(f)
	with open('./cache/uspto/dictionary.msgpack', 'rb') as f:
		uspto_dict = msgpack.unpack(f)

	# Calculate the year indices for the years of interest
	year_indices = [i - start_year for i in years_of_interest]
	curr_year_index = year_indices[4]

	# Create networkx graph from matrix
	a = adj_matrices[curr_year_index]
	G = nx.from_numpy_matrix(a, create_using=nx.DiGraph())

	# Calculate the degrees for each category in the adjacency matrices
	unnormalized_degrees, normalized_degrees = calculate_degrees(adj_matrices, vectors)

	# Draw the graph using networkx
	pos = nx.spring_layout(G)
	sizes = [row[1] for row in unnormalized_degrees[curr_year_index]] # In-degree for each node
	reverse_uspto_dict = {v: k for k, v in uspto_dict.iteritems()}
	usptos = [reverse_uspto_dict[i] for i in range(len(a))] # Uspto category for each index in adjacency matrix
	# Generate colors and map each ipc8 category to a distinct color
	cmap = plt.cm.jet
	colors = cmap(np.linspace(0, 1, 8))
	ipcs = [cw_dict[uspto] for uspto in usptos] # Ipc category for each index in adjacency matrix
	ipc_to_color = {}
	reverse_cw_dict = {v: k for k, v in cw_dict.iteritems()}
	for i, ipc in enumerate(reverse_cw_dict.iterkeys()):
		ipc_to_color[ipc] = colors[i]
	ipc_colors = [ipc_to_color[ipc] for ipc in ipcs]

	# Draw the graph using networkx
	nx.draw(G, pos=pos, with_labels=False, node_color=ipc_colors, node_size=[s * 25 for s in sizes], width=0.05, arrowsize=3, cmap=cmap)  # networkx draw()

	# Create a legend displaying the mapping from ipc to a color
	patchList = []
	visited_ipc = set() # Only want to map each ipc once to a color
	for i in range(len(ipcs)):
			if ipcs[i] in visited_ipc:
				continue
			else:
				data_key = mpatches.Patch(color=ipc_colors[i], label=ipcs[i])
		        patchList.append(data_key)
		        visited_ipc.add(ipcs[i])

	plt.legend(handles=patchList, loc='upper right')

	plt.show()

# Graphs the heatmap for the years of interest
def graph_heatmap(adj_matrices, start_year, years_of_interest):
	# Calculate the year indices for the years of interest
	year_indices = [i - start_year for i in years_of_interest]
	curr_year_index = year_indices[3]

	a = matrices[curr_year_index]
	# for i in range(len(a)):
	# 	a[i,i] = 0
	plt.imshow(a, cmap='hot', interpolation='nearest')
	plt.colorbar()
	plt.show()

# Aggregate the matrices or vectors for every x years
# ex. aggregate every 5 years worth of data
def aggregate_years(array, years_per_aggregate):
	return [sum(array[i:i+5]) for i in range(0, len(array), years_per_aggregate)]

# First load the serialized vectors and matrices
vectors, matrices, cat_dict = load_network(network_to_use)

# calculate_eigenvector_centrality(network_to_use, matrices, years_per_aggregate)

# Graph the networks for some years
graph_network(matrices, vectors, start_year, years_to_graph)

# Graph heatmap
# graph_heatmap(matrices, start_year, years_to_graph)

# TODO:
# Work on graphing
# Graphviz
# Color edge based on intensity
# May need to halve the weights of the diagonals of each matrix, but only if degrees is called for matrix
# -does not affect in and out degree calls


# # TEST
# d = np.matrix([[2,0,0],
# 			   [0,0,0],
# 			   [5,0,0]])
# G = nx.from_numpy_matrix(d, create_using=nx.DiGraph())
# print d
# print G

# print G.in_degree([0,1,2], weight='weight')
# print G.out_degree([0,1,2], weight='weight')

# # Draw the graph using networkx
# pos = nx.spring_layout(G)
# cmap = plt.cm.jet
# values = dict(nx.degree(G, weight='weight')) # In-degree for each node
# print values
# nx.draw(G, pos=pos, with_labels=True, node_color=[v * 25 for v in values.values()], node_size=[v * 25 for v in values.values()], cmap=cmap)  # networkx draw()
# write_dot(G,'graph.dot')

# # Set color bar
# sm = plt.cm.ScalarMappable(cmap=cmap)
# sm._A = []
# sm.set_clim(vmin=min(values.values()), vmax=max(values.values()))
# plt.colorbar(sm, shrink=0.7)
# print(max(values))
# print(min(values))

# plt.show()  # pyplot draw()

