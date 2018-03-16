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
network_to_use = 'ipc8' # uspto or ipc108 or ipc8
years_per_aggregate = 5 # number of years of data in each matrix/vector
normalization_choice = 'norm1' # Normalization choice for network degrees (norm1 or norm2)
ipc8_to_category_name = {
	'a': 'Human Necessities',
	'b': 'Performing Operations',
	'c': 'Chemistry; Metallurgy',
	'd': 'Textiles; Paper',
	'e': 'Fixed Constructions',
	'f': 'Mech Eng; Lighting; Heating; Weapons',
	'g': 'Physics',
	'h': 'Electricity'
}

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
	normalized_degrees_1 = [np.zeros((n, 2)) for i in range(num_years)] # Create an Nx2 vector for each year
	normalized_degrees_2 = [np.zeros((n, 2)) for i in range(num_years)] # Create an Nx2 vector for each year

	# Calculate the in and out degrees for each category for each year
	# Two types of normalization:
	# 1. Divide each in/out degree for each category by the total number of patents in that category
	# 2. Divide each in/out degree by the total number of in-degrees for that year
	for year, matrix in enumerate(adj_matrices):
		# Create a networkx multi directed graph
		G = nx.from_numpy_matrix(matrix, create_using=nx.DiGraph())
		# Track the total number of in-degrees
		total_in_degrees = 0
		# Iterate through each category
		for category in range(len(vectors[year])):
			# Find the degrees for each category
			total_patents = vectors[year][category]
			in_degree = G.in_degree(category, weight='weight')
			out_degree = G.out_degree(category, weight='weight')
			total_in_degrees += in_degree

			# Find normalized values
			if total_patents == 0:
				normalized_out_degree = 0
				normalized_in_degree = 0
			else: 
				normalized_out_degree = float(out_degree)/total_patents
				normalized_in_degree = float(in_degree)/total_patents

			# Write degree values into the list of vectors
			unnormalized_degrees[year][category] = [out_degree, in_degree]
			normalized_degrees_1[year][category] = [normalized_out_degree, normalized_in_degree]

		# Iterate through each category again to calculate the second normalization using total number of in-degrees
		for category in range(len(vectors[year])):
			out_degree = unnormalized_degrees[year][category][0]
			in_degree = unnormalized_degrees[year][category][0]

			# Find normalized values
			if total_in_degrees == 0:
				normalized_out_degree = 0
				normalized_in_degree = 0
			else: 
				normalized_out_degree = float(out_degree)/total_in_degrees
				normalized_in_degree = float(in_degree)/total_in_degrees

			# Write normalized degree values into the second normalization array
			normalized_degrees_2[year][category] = [normalized_out_degree, normalized_in_degree]

	return unnormalized_degrees, normalized_degrees_1, normalized_degrees_2

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
		# Add 2 entries to the rankings_by_year array (one for the patent name, other for eigenvalue)
		rankings_by_year.append((curr_year, [x[0] for x in rankings]))
		rankings_by_year.append((curr_year, [x[1] for x in rankings]))

	# Write into a csv file
	years = [x[0] for x in rankings_by_year] # Extract the years to form the first row
	rankings = [x[1] for x in rankings_by_year] # Extract rankings
	transposed_rankings = zip(*rankings)
	f_name = './cache/' + network_to_use + '/eigenvector_rankings_' + str(years_per_aggregate) + 'year_aggregates.csv'
	with open(f_name, 'wb') as f:
		writer = csv.writer(f)
		writer.writerow(years)
		for row in transposed_rankings:
			writer.writerow(row)

	# Plot the rankings for each category over time if network_to_use is ipc8
	if network_to_use == 'ipc8':
		cmap = plt.cm.jet
		colors = cmap(np.linspace(0, 1, 8))
		rankings = {} # Dictionary where each key corresponds to a ipc8 letter and the value is a list of rankings overtime
		years = [] # Years to be used as x axis
		# Add rankings year by year into the rankings dictionary
		for i in range(0, len(rankings_by_year), 2):
			years.append(rankings_by_year[i][0])
			ipcs_by_ranking = rankings_by_year[i][1]
			for i, ipc in enumerate(ipcs_by_ranking):
				if ipc not in rankings:
					rankings[ipc] = [i+1]
				else:
					rankings[ipc].append(i+1)
		# Plot the rankings for each ipc over time and create a legend
		patchList = []
		for i, key in enumerate(rankings.iterkeys()):
			# Plot the ranking for this ipc8
			plt.plot(years, rankings[key], color=colors[i])
			# Add data key for this ipc8 to legend
			data_key = mpatches.Patch(color=colors[i], label=ipc8_to_category_name[key])
			patchList.append(data_key)
		plt.title('Rankings of Eigenvector Centrality Over Time')
		plt.legend(handles=patchList, loc='upper right', title='ipc8 categories', fontsize='x-small', bbox_to_anchor=(1.5, 0.65),
          fancybox=True, shadow=True)
		plt.ylim(9, 0) # Y-axis is decreasing because higher ranked categories with lower values should be on top
		plt.ylabel('Centrality Ranking')
		plt.xlabel('Year')
		# plt.show()

		# Save the plot to file
		plt.show(block=False)
		name = 'eigenvector_centrality_rankings_over_time'
		plt.savefig('./generated_plots/' + name + '.png', bbox_inches='tight')

	return rankings_by_year

# Calculate the Pagerank centrality for the network
def calculate_pagerank_centrality(network_to_use, adj_matrices, years_per_aggregate):
	# List to be populated with pagerank measures for each year
	rankings_by_year = []

	# Aggregate the years of the matrices together
	aggregated_matrices = aggregate_years(adj_matrices, years_per_aggregate)

	# Loop through each matrix and calculate the eignvector, then rank each patent by weight
	for i, matrix in enumerate(aggregated_matrices):
		curr_year = start_year + i * years_per_aggregate
		# Convert matrix to a MultiDiGraph
		G = nx.from_numpy_matrix(matrix, create_using=nx.MultiDiGraph())
		# Find eigenvector centrality
		centrality = nx.pagerank_numpy(G, weight='weight')
		# Maps matrix index value to a uspto value
		reverse_uspto_dict = {v: k for k, v in cat_dict.iteritems()}
		# Create a list of rankings, where the most central patents are first
		rankings = [[reverse_uspto_dict[k], v] for k, v in centrality.iteritems()]
		rankings = sorted(rankings, key = lambda x: x[1])[::-1] # Sort by value
		# Add 2 entries to the rankings_by_year array (one for the patent name, other for pagerank value)
		rankings_by_year.append((curr_year, [x[0] for x in rankings]))
		rankings_by_year.append((curr_year, [x[1] for x in rankings]))
	print rankings

	# Write into a csv file
	years = [x[0] for x in rankings_by_year] # Extract the years to form the first row
	rankings = [x[1] for x in rankings_by_year] # Extract rankings
	transposed_rankings = zip(*rankings)
	f_name = './cache/' + network_to_use + '/pagerank_rankings_' + str(years_per_aggregate) + 'year_aggregates.csv'
	with open(f_name, 'wb') as f:
		writer = csv.writer(f)
		writer.writerow(years)
		for row in transposed_rankings:
			writer.writerow(row)

	# Plot the rankings for each category over time if network_to_use is ipc8
	if network_to_use == 'ipc8':
		cmap = plt.cm.jet
		colors = cmap(np.linspace(0, 1, 8))
		pagerank_values_overtime = {} # {key: ipc8 letter, value: list of pagerank values overtime}
		years = [] # Years to be used as x axis
		# Add Pagerank values year by year into the pagerank dictionary
		for i in range(0, len(rankings_by_year), 2):
			years.append(rankings_by_year[i][0])
			ipcs_by_ranking = rankings_by_year[i][1]
			pagerank_values = rankings_by_year[i+1][1]
			for i, ipc in enumerate(ipcs_by_ranking):
				if ipc not in pagerank_values_overtime:
					pagerank_values_overtime[ipc] = [pagerank_values[i] * 100]
				else:
					pagerank_values_overtime[ipc].append(pagerank_values[i] * 100)
		# Plot the Pagerank values for each ipc over time and create a legend
		patchList = []
		for i, key in enumerate(pagerank_values_overtime.iterkeys()):
			# Plot the ranking for this ipc8
			plt.plot(years, pagerank_values_overtime[key], color=colors[i])
			# Add data key for this ipc8 to legend
			data_key = mpatches.Patch(color=colors[i], label=ipc8_to_category_name[key])
			patchList.append(data_key)
		plt.title('Pagerank Centrality Values Over Time')
		plt.legend(handles=patchList, loc='upper right', title='ipc8 categories', fontsize='x-small', bbox_to_anchor=(1.5, 0.65),
          fancybox=True, shadow=True)
		plt.ylabel('Pagerank Value')
		plt.xlabel('Year')
		# plt.show()

		# Save the plot to file
		plt.show(block=False)
		name = 'pagerank_centrality_values_over_time'
		plt.savefig('./generated_plots/' + name + '.png', bbox_inches='tight')

	return rankings_by_year

# Graph the networks for the years of interest
# Nodes will be colored based on their ipc8 category and sized based on their unnormalized in-degrees
def graph_network(network_to_use, adj_matrices, vectors):
	# Depending on the network_to_use we are graphing, the crosswalk dictionary file we load will differ
	# network_to_use = uspto
	if network_to_use == 'uspto':
		with open('./cache/uspto/cw_dictionary_to_ipc8.msgpack', 'rb') as f:
			cw_dict = msgpack.unpack(f)
		with open('./cache/uspto/dictionary.msgpack', 'rb') as f:
			cat_dict = msgpack.unpack(f)
	elif network_to_use == 'ipc108':
		with open('./cache/ipc108/dictionary.msgpack', 'rb') as f:
			cat_dict = msgpack.unpack(f)
		# Generate cw_dict mapping ipc108 to ipc8 
		cw_dict = {}
		for key in cat_dict:
			if key not in cw_dict:
				cw_dict[key] = key[0]
	elif network_to_use == 'ipc8':
		with open('./cache/ipc8/dictionary.msgpack', 'rb') as f:
			cat_dict = msgpack.unpack(f)
		# Generate cw_dict mapping ipc8 to ipc8 
		cw_dict = {}
		for key in cat_dict:
			if key not in cw_dict:
				cw_dict[key] = key

	# Calculate the year indices for the years of interest
	year_indices = [i - start_year for i in years_to_graph]

	# Loop through each of the years of interest and generate its network graph
	for year_index in year_indices:
		# Current year
		curr_year = start_year + year_index

		# Create networkx graph from matrix
		a = adj_matrices[year_index]
		G = nx.from_numpy_matrix(a, create_using=nx.DiGraph())

		# Calculate the degrees for each category in the adjacency matrices
		unnormalized_degrees, normalized_degrees_1, normalized_degrees_2 = calculate_degrees(adj_matrices, vectors)

		# Choose normalized degrees based on normalization choice
		if normalization_choice == 'norm1':
			normalized_degrees = normalized_degrees_1
		elif normalization_choice == 'norm2':
			normalized_degrees = normalized_degrees_2

		# Draw the graph using networkx
		plt.figure()
		pos = nx.random_layout(G)
		sizes = [row[1] for row in normalized_degrees[year_index]] # In-degree for each node
		reverse_cat_dict = {v: k for k, v in cat_dict.iteritems()}
		categories = [reverse_cat_dict[i] for i in range(len(a))] # Category for each index in adjacency matrix
		# Generate colors and map each ipc8 category to a distinct color
		cmap = plt.cm.jet
		colors = cmap(np.linspace(0, 1, 8))
		ipcs = [cw_dict[cat] for cat in categories] # Ipc8 category for each index in adjacency matrix
		ipc_to_color = {}
		reverse_cw_dict = {v: k for k, v in cw_dict.iteritems()}
		for i, ipc in enumerate(reverse_cw_dict.iterkeys()):
			ipc_to_color[ipc] = colors[i]
		ipc_colors = [ipc_to_color[ipc] for ipc in ipcs]

		# Draw the graph using networkx
		plt.title('Network of patents by ' + network_to_use + ' in ' + str(curr_year))
		nx.draw(G, pos=pos, with_labels=False, node_color=ipc_colors, node_size=[s * 10000 for s in sizes], width=0.1, arrowsize=6, cmap=cmap)  # networkx draw()

		# Create a legend displaying the mapping from ipc to a color
		patchList = []
		visited_ipc = set() # Only want to map each ipc once to a color
		for i in range(len(ipcs)):
				if ipcs[i] in visited_ipc:
					continue
				else:
					data_key = mpatches.Patch(color=ipc_colors[i], label=ipc8_to_category_name[ipcs[i]])
			        patchList.append(data_key)
			        visited_ipc.add(ipcs[i])

		plt.legend(handles=patchList, loc='upper right', title='ipc8 categories', fontsize='xx-small', bbox_to_anchor=(1.05, 0.03),
	          fancybox=True, shadow=True, ncol=4)

		# Save the plot to file
		plt.show(block=False)
		fpath = './generated_plots/network_graphs/' + normalization_choice + '/' + network_to_use + '/'
		fname = network_to_use + ' ' + str(curr_year) + '.png'
		plt.savefig(fpath + fname, bbox_inches='tight')

# Graphs the heatmap for the years of interest
def graph_heatmap(adj_matrices):
	# Calculate the year indices for the years of interest
	year_indices = [i - start_year for i in years_to_graph]
	curr_year_index = year_indices[4]

	a = adj_matrices[curr_year_index]
	# for i in range(len(a)):
	# 	a[i,i] = a[i,i] / 2
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

calculate_pagerank_centrality(network_to_use, matrices, years_per_aggregate)

# Graph the networks for some years
# graph_network(network_to_use, matrices, vectors)

# Graph heatmap
# graph_heatmap(matrices)

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

