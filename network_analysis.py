import pandas as pd
import numpy as np
import msgpack
import msgpack_numpy as m
import networkx as nx
import matplotlib.pyplot as plt
import os

from networkx.drawing.nx_agraph import write_dot

##### Script for analyzing the network of patents (in/out degrees, eigenvector centrality, etc.) and graphing #####

# Patch msgpack_numpy so that msgpack can serialize numpy objects
m.patch()

# Initialize variables
start_year = 1858 # Default: 1858
end_year = 2015 # Default 2015
year_gap = 10
years_to_graph = [1860, 1880, 1900, 1920, 1940, 1960, 1980, 2000]
network_to_load = 'uspto' # uspto or ipc
aggregate_years = 5 # number of years of data in each matrix/vector

# Loads the vectors and adjacency matrixes
def load_network(network_to_load):
	if network_to_load == 'uspto':
		with open('uspto_vectors.msgpack', 'rb') as f:
				vectors = msgpack.unpack(f)
		with open('uspto_matrices.msgpack', 'rb') as f:
				matrices = msgpack.unpack(f)
		with open('uspto_dictionary.msgpack', 'rb') as f:
				cat_dict = msgpack.unpack(f)
	elif network_to_load == 'ipc':
		with open('ipc_vectors.msgpack', 'rb') as f:
				vectors = msgpack.unpack(f)
		with open('ipc_matrices.msgpack', 'rb') as f:
				matrices = msgpack.unpack(f)
		with open('ipc_dictionary.msgpack', 'rb') as f:
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
def calculate_eigenvector_centrality(adj_matrices):
	# List to be populated with centrality measures for each year

	# Loop through each year
	# for year, matrix in enumerate(adj_matrices):
	# 	# Create a networkx multi directed graph
	# 	G = nx.from_numpy_matrix(matrix, create_using=nx.MultiDiGraph())
	# 	# Find eigenvector centrality
	# 	centrality = nx.eigenvector_centrality_numpy(G, weight='weight')
	# 	print year
	# 	print centrality
	G = nx.from_numpy_matrix(adj_matrices[145], create_using=nx.MultiDiGraph())
	# Find eigenvector centrality
	centrality = nx.eigenvector_centrality_numpy(G, weight='weight')
	print centrality
	print max(centrality.values())
	max_centrality_index = max(centrality, key=centrality.get)
	print max_centrality_index
	reverse_uspto_dict = {v: k for k, v in cat_dict.iteritems()}
	print reverse_uspto_dict[max_centrality_index]

# Graph the networks for the years of interest
def graph_network(adj_matrices, start_year, years_of_interest, degrees):
	# Calculate the year indices for the years of interest
	year_indices = [i - start_year for i in years_of_interest]
	curr_year_index = year_indices[3]

	# Create networkx graph from matrix
	G = nx.from_numpy_matrix(adj_matrices[curr_year_index], create_using=nx.DiGraph())

	edges = G.edges()
	print edges # FIND WEIGHTS OF EDGES, THIS DOES NOT WORK

	# Draw the graph using networkx
	pos = nx.spring_layout(G)
	cmap = plt.cm.jet
	values = [row[1] for row in degrees[curr_year_index]] # In-degree for each node
	nx.draw(G, pos=pos, with_labels=False, node_color=values, node_size=[v * 25 for v in values], width=0.05, arrowsize=3, cmap=cmap)  # networkx draw()

	# Set color bar
	sm = plt.cm.ScalarMappable(cmap=cmap)
	sm._A = []
	sm.set_clim(vmin=min(values), vmax=max(values))
	plt.colorbar(sm, shrink=0.7)
	print(max(values))
	print(min(values))

	plt.show()  # pyplot draw()

# Aggregate the matrices or vectors for every x years
# ex. aggregate every 5 years worth of data
def aggregate_years(array, years):
	return [sum(array[i:i+5]) for i in range(0, len(array), years)]

# First load the serialized vectors and matrices
vectors, matrices, cat_dict = load_network(network_to_load)

# Calculate the degrees for each category in the adjacency matrices
# unnormalized_degrees, normalized_degrees = calculate_degrees(matrices, vectors)
# print normalized_degrees
# # print unnormalized_degrees

calculate_eigenvector_centrality(matrices)
# np.set_printoptions(threshold=np.inf)

# Graph the networks for some years
# graph_network(matrices, start_year, years_to_graph, unnormalized_degrees)

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