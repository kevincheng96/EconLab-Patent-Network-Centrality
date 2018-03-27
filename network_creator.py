#!python2

import pandas as pd
import numpy as np
import msgpack
import msgpack_numpy as m
import os

##### Script for extracting data from csv files and constructing the vectors and adjacency matrices #####

# Patch msgpack_numpy so that msgpack can serialize numpy objects
m.patch()

# Initialize variables
start_year = 1835 # Default: 1835
end_year = 2015 # Default 2015
year_gap = 10
num_classes = 'few' # 'many' = 108 classes, 'few' = 8 main classes

# Load in the patent data
# Dictionary of format: {patnum: {fyear: int, main_uspto: int}}
def retrieve_patent_data():
	# Check if data is already stored in directory
	if os.path.isfile('./cache/patents.msgpack'):
		print 'loading data'
		with open('./cache/patents.msgpack', 'rb') as f:
			patents = msgpack.unpack(f)
		print 'done loading data'
	else:
		# Get the filing year for each patent
		df = pd.read_csv('./data/patents_fyear_iyear.csv')
		fyears = df[['patnum','fyear','iyear']] # Dataframe containing three columns: patnum, fyear, and iyear
		fyears.set_index('patnum')
		print 'loaded first'

		# Get the main category for each patent
		df = pd.read_csv('./data/patents_uspto_categories_11fix.csv')
		categories = df[['patnum','main_uspto']] # Dataframe containing two columns: patnum and main_uspto
		print categories['main_uspto']
		categories['main_uspto'] = categories['main_uspto'].apply(lambda x: str(x).split('/')[0]) # ex. '301/105' -> '301'  
		categories.set_index('patnum')
		print 'loaded second'

		# Join the two dataframes together 
		# Format = patnum fyear iyear main_uspto
		print 'joining'
		patents = fyears.merge(categories)
		print patents
		print patents.shape

		# Convert dataframe into a dictionary of format: {patnum: {fyear: int, iyear: int, main_uspto: int}}
		patents.set_index('patnum', drop=True, inplace=True)
		print 'set index'
		d = patents.to_dict(orient='index')

		# Save dictionary into a serialized file
		print 'dumping'
		with open('./cache/patents.msgpack', 'wb') as f:
			print 'still dumping'
			msgpack.pack(d, f)
		print 'done dumping'
		patents = d

	return patents

# Create network category dictionary: {main_uspto: index of category in adj matrix}
def create_uspto_dict(patents, valid_usptos):
	# First find out how many distinct categories there are
	uspto_set = set() # Set containing all main_uspto values
	uspto_dict = {} # Dictionary mapping each main_uspto to its index value in adjacency matrix

	for d in patents.itervalues():
		if d['main_uspto'] in valid_usptos:
			uspto_set.add(d['main_uspto'])

	counter = 0
	for item in uspto_set:
		uspto_dict[item] = counter
		counter += 1

	print uspto_set

	return uspto_dict

# Create vector and adjacency matrix ###### CREATE A LIST OF ADJACENCY MATRICES SPANNING n YEARS EACH
# fn = filename of csv, patents = dictionary of patents, fyear_gap = maximum allowed years between cited and citing patent
def create_vector_and_matrix(patents, start_year, end_year, fyear_gap):
	# Read from the crosswalk file to find non-faulty uspto values to keep later on
	df = pd.read_csv('./data/cw_uspto_ipc_cpc.csv')
	df_usptos = df[['uspto']] # Crosswalk dataframe containing two columns: uspto and ipc
	
	# Construct the set of all non-faulty uspto values
	valid_usptos = set() # Set of all correct usptos
	for i, row in df_usptos.iterrows():
		valid_usptos.add(str(row['uspto']))

	# Dictionary mapping each main_uspto to its index value in adjacency matrix
	uspto_dict = create_uspto_dict(patents, valid_usptos)
	n = len(uspto_dict)

	# Initialize the list of vectors and matrices
	num_years = end_year - start_year
	# Vector keeps track of how many patents cited each category over the next num_years
	vectors = [np.zeros((n, 1)) for i in range(num_years)] # Create an Nx1 vector for each year
	# Adjacency matrix keeps track of which categories cited which categories (directed graph)
	# (row i, col j) = category at row i points to (cited) category at column j
	matrices = [np.zeros((n, n)) for i in range(num_years)] # Create an NxN adjacency matrix for each year

	# Find all csv files in cit_received directory
	files = [file for file in os.listdir('./data/cit_received') if file.endswith(".csv")]

	counter1 = 0
	# Read in each file in directory
	for f in files:
		print './data/cit_received/' + f
		# Read csv file in chunks (takes too much memory to load all at once)
		# Only the first csv file has headers
		if f == 'patents_cit_received_part1.csv':
			reader = pd.read_csv('./data/cit_received/' + f, chunksize=50, error_bad_lines=False, warn_bad_lines=True)
		else:
			reader = pd.read_csv('./data/cit_received/' + f, chunksize=50, header=None, error_bad_lines=False, warn_bad_lines=True)
		counter = 0
		counter1 += 1

		# Read each chunk
		for chunk in reader:
			if counter % 1000 == 0:
				print str(counter1) + '"th file, step: ' + str(counter)
			counter += 1
			# Read in chunk as a dataframe
			df = chunk
			df.set_index(df.columns[0], drop=True, inplace=True)
			df = df.dropna(axis=1, how='all') # Drop all empty columns

			# Iterate through each patent in the file and all the patents that cite it
			for patnum, row in df.iterrows():
				fyear, uspto = patents[patnum]['fyear'], patents[patnum]['main_uspto']
				# Skip the current patent if it has a faulty uspto number
				if uspto not in valid_usptos:
					continue
				# If fyear is NaN, use the iyear instead
				if pd.isnull(fyear):
					iyear = patents[patnum]['iyear']
					if not pd.isnull(iyear):
						fyear = int(iyear)
					else:
						continue
				else:
					fyear = int(fyear)
				i = fyear - start_year # index of this year's matrix inside the adj_matrices list
				j = uspto_dict[uspto] # index in adjacency matrix for current patent's category
				# Add entry to vector saying that category j has a patent filed this year
				vectors[i][j] += 1
				# Iterate through each other patent that cites the current patnum
				for cit_by in row:
					# Move on to next row if cit_by is NaN
					if pd.isnull(cit_by):
						break
					else:
						cit_fyear, cit_uspto = patents[cit_by]['fyear'], patents[cit_by]['main_uspto']
						# Skip the current patent if it has a faulty uspto number
						if cit_uspto not in valid_usptos:
							continue
						# If the fyear for this citing patent is null, use iyear instead
						if pd.isnull(cit_fyear):
							cit_iyear = patents[cit_by]['iyear']
							if not pd.isnull(cit_iyear):
								cit_fyear = int(cit_iyear)
							else:
								continue
						# If citing patent is filed more than fyear_gap years after the cited patent, ignore
						if cit_fyear - fyear  - 1 > fyear_gap or cit_fyear - fyear <= 0:
							continue
						else:
							# Add entry into adjacency matrix
							k = uspto_dict[cit_uspto] # adj matrix index for the category of patent that cited this patent
							# A directed edge goes from category k (citing patent) to category j (cited patent) in time i
							matrices[i][k,j] += 1

	# Save vectors and matrices into serialized files
	print 'dumping vectors'
	with open('./cache/uspto/vectors.msgpack', 'wb') as f:
		msgpack.pack(vectors, f)
	print 'dumping matrices'
	with open('./cache/uspto/matrices.msgpack', 'wb') as f:
		msgpack.pack(matrices, f)
	print 'dumping uspto_dict'
	with open('./cache/uspto/dictionary.msgpack', 'wb') as f:
		msgpack.pack(uspto_dict, f)
	print 'done dumping'

	print vectors, matrices

# Converts the adjacency matrix and vector from 430+ uspto categories to less categories using the crosswalk data
# Note: Converts to ipc8 if num_classes=='few' and ipc108 if num_classes=='many'
def apply_crosswalk(num_classes):
	# Read in existing data
	with open('./cache/uspto/vectors.msgpack', 'rb') as f:
			uspto_vectors = msgpack.unpack(f)
	with open('./cache/uspto/matrices.msgpack', 'rb') as f:
			uspto_matrices = msgpack.unpack(f)
	with open('./cache/uspto/dictionary.msgpack', 'rb') as f:
			uspto_dict = msgpack.unpack(f)

	# Read from the crosswalk file
	df = pd.read_csv('./data/cw_uspto_ipc_cpc.csv')
	cw = df[['uspto','ipc']] # Crosswalk dataframe containing two columns: uspto and ipc
	# Converts either to 107 classes or 8 main classes
	if num_classes == 'many':
		cw['ipc'] = cw['ipc'].apply(lambda x: str(x)[:3]) # ex. 'b30b -> b30'
	elif num_classes == 'few':
		cw['ipc'] = cw['ipc'].apply(lambda x: str(x)[:1]) # ex. 'b30b -> b'    
	
	# Construct a dictionary for the crosswalk that maps each uspto to an ipc
	cw_dict = {}
	for i, row in cw.iterrows():
		uspto, ipc = str(row['uspto']), str(row['ipc'])
		cw_dict[uspto] = ipc

	# Construct a dictionary for ipc that maps each ipc value to its index value in the new matrices and vectors
	ipcs = set() # Set containing all ipc values
	ipc_dict = {} # Dictionary mapping each ipc to its index value in adjacency matrix

	for i, row in cw.iterrows():
		ipcs.add(str(row['ipc']))

	counter = 0
	for item in ipcs:
		ipc_dict[item] = counter
		counter += 1

	# Now, reconstruct the matrices and vectors using the crosswalk
	n = len(ipc_dict) # n = number of distinct ipc's
	ipc_vectors = [np.zeros((n, 1)) for i in range(len(uspto_vectors))] # Create an Nx1 vector for each year
	ipc_matrices = [np.zeros((n, n)) for i in range(len(uspto_matrices))] # Create an NxN adjacency matrix for each year
	reverse_uspto_dict = {v: k for k, v in uspto_dict.iteritems()} # Dictionary mapping each index value to a uspto class (inverse of uspto_dict)

	# Construct each matrix/vector year by year
	for i in range(len(uspto_matrices)): # i is index for each year
		for j in range(len(uspto_matrices[i])): # j is index for each category within each year
			# Want to find the uspto for each index in the original matrix, then map that to a ipc using the cw_dict
			# Then, transfer that value to the ipc_matrices/vectors by finding the ipc's index in ipc_dict
			uspto = reverse_uspto_dict[j]
			# Skip current patent if it has a faulty id
			if uspto not in cw_dict:
				continue
			else:
				ipc = cw_dict[uspto]
			ipc_index = ipc_dict[ipc]

			# Transfer value from uspto_matrix/vector to ifc_matrix/vector
			ipc_vectors[i][ipc_index] += uspto_vectors[i][j]
			# Loop through each column k in row j of this matrix
			for k in range(len(uspto_matrices[i][j])):
				uspto_2 = reverse_uspto_dict[k]
				# Skip current patent if it has a faulty id
				if uspto_2 not in cw_dict:
					continue
				else:
					ipc_2 = cw_dict[uspto_2]
				ipc_index_2 = ipc_dict[ipc_2]
				ipc_matrices[i][ipc_index, ipc_index_2] += uspto_matrices[i][j, k]

	# Save vectors and matrices into serialized files
	if num_classes == 'many':
		prefix = 'ipc108'
	elif num_classes == 'few':
		prefix = 'ipc8'

	with open('./cache/' + prefix + '/vectors.msgpack', 'wb') as f:
		msgpack.pack(ipc_vectors, f)
	with open('./cache/' + prefix + '/matrices.msgpack', 'wb') as f:
		msgpack.pack(ipc_matrices, f)
	with open('./cache/' + prefix + '/dictionary.msgpack', 'wb') as f:
		msgpack.pack(ipc_dict, f)
	with open('./cache/uspto/' + 'cw_dictionary_to_' + prefix + '.msgpack', 'wb') as f:
		msgpack.pack(cw_dict, f)
	print 'done dumping'

	return

patents = retrieve_patent_data() # Dictionary of format: {patnum: {fyear: int, main_uspto: int}}

create_vector_and_matrix(patents, start_year, end_year, year_gap)

apply_crosswalk(num_classes)


# TODO:
# if fyear is missing, use iyear (Done)
# crosswalk, skip over bad uspto numbers (Done)
# create 5 year aggregate (Done)
# run eigenvector centrality measure on these aggregates and save output (year, ranked categories, centrality measure) (Dones)
# Output CSV (Done)
# Heatmap (Done)
# Category web network where the same ipc_8 patents have the same color (Done)
# Normalize by the total number of in-degrees for each year instead (Done)
# Graph rankings over time (y-axis: rank) (Done)
# Figure out why plots for earlier years are blank (Apparently the matrices for earlier years are empty even though vectors are not) (Done)
# Graph pagerank rankings (y-axis: 0-100 to show intensity) (Done)
# Find a way to graph norm1 (since the sizes are disproportionate)
# Graph heatmaps and fix diagonals (Done)
# Generate CSV for centrality ranking/page rank plots (Done)
# Generate CSV of rankings for in-degrees for ipc108 (year, in-degree, ipc108) (Done)
# Aggregate by 10 years, then do plots  (Done)
# Add ipc labels for ipc8 heatmap
# For in-degree CSV, add in ipc108 name as a column
