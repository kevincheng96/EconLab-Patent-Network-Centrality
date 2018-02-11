import pandas as pd
import numpy as np
import msgpack
import msgpack_numpy as m
import networkx as nx
import matplotlib.pyplot as plt
import os

##### Script for analyzing the network of patents (in/out degrees, eigenvector centrality, etc.) and graphing #####

# Patch msgpack_numpy so that msgpack can serialize numpy objects
m.patch()

# Initialize variables
start_year = 1858 # Default: 1858
end_year = 2015 # Default 2015
year_gap = 10
years_to_graph = [1860, 1880, 1900, 1920, 1940, 1960, 1980, 2000]
network_to_load = 'uspto' # uspto or ipc

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
	for year, matrix in enumerate(adj_matrices):
		# Create a networkx multi directed graph
		G = nx.from_numpy_matrix(matrix, create_using=nx.MultiDiGraph())
		# Find eigenvector centrality
		centrality = nx.eigenvector_centrality_numpy(G, weight='weight')
		print year

# Graph the networks for the years of interest
def graph_network(adj_matrices, start_year, years_of_interest, normalized_degrees):
	# Calculate the year indices for the years of interest
	year_indices = [i - start_year for i in years_of_interest]
	curr_year_index = year_indices[3]
	G = nx.from_numpy_matrix(adj_matrices[curr_year_index], create_using=nx.DiGraph())
	pos = nx.spring_layout(G)
	cmap = plt.cm.jet
	values = [row[1] for row in normalized_degrees[curr_year_index]] # In-degree
	print values
	nx.draw(G, pos=pos, with_labels=False, node_color=values, node_size=50, cmap=cmap)  # networkx draw()
	plt.show()  # pyplot draw()

# First load the serialized vectors and matrices
vectors, matrices, cat_dict = load_network(network_to_load)

# Calculate the degrees for each category in the adjacency matrices
unnormalized_degrees, normalized_degrees = calculate_degrees(matrices, vectors)
print normalized_degrees
# # print unnormalized_degrees

# calculate_eigenvector_centrality(matrices)
# np.set_printoptions(threshold=np.inf)

# Graph the networks for some years
graph_network(matrices, start_year, years_to_graph, normalized_degrees)

# TODO:
# Work on graphing
