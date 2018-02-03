import pandas as pd
import numpy as np
import msgpack
import msgpack_numpy as m
import networkx as nx
import os

# Patch msgpack_numpy so that msgpack can serialize numpy objects
m.patch()

# Loads the vectors and adjacency matrixes
def load_network():
	with open('vectors.msgpack', 'rb') as f:
			vectors = msgpack.unpack(f)
	with open('matrices.msgpack', 'rb') as f:
			matrices = msgpack.unpack(f)
	with open('category_dictionary.msgpack', 'rb') as f:
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
		G = nx.from_numpy_matrix(matrix, create_using=nx.MultiDiGraph())
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

# First load the serialized vectors and matrices
vectors, matrices, cat_dict = load_network()

# Calculate the degrees for each category in the adjacency matrices
unnormalized_degrees, normalized_degrees = calculate_degrees(matrices, vectors)
print unnormalized_degrees

