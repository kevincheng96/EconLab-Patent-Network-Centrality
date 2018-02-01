import pandas as pd
import numpy as np
import msgpack
import msgpack_numpy as m
import os

# Patch msgpack_numpy so that msgpack can serialize numpy objects
m.patch()

# Initialize variables
start_year = 1858 # Default: 1858
end_year = 2014 # Default 2014
year_gap = 10

# Load in the patent data
# Dictionary of format: {patnum: {fyear: int, main_uspto: int}}
def retrieve_patent_data():
	# Check if data is already stored in directory
	if os.path.isfile('patents.msgpack'):
		print 'loading data'
		with open('patents.msgpack', 'rb') as f:
			patents = msgpack.unpack(f)
		print 'done loading data'
	else:
		# Get the filing year for each patent
		df = pd.read_csv('./data/patents_fyear_iyear.csv')
		fyears = df[['patnum','fyear']] # Dataframe containing two columns: patnum and fyear
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
		# Format = patnum fyear main_uspto
		print 'joining'
		patents = fyears.merge(categories)
		print patents
		print patents.shape

		# Convert dataframe into a dictionary of format: {patnum: {fyear: int, main_uspto: int}}
		patents.set_index('patnum', drop=True, inplace=True)
		print 'set index'
		d = patents.to_dict(orient='index')

		# Save dictionary into a serialized file
		print 'dumping'
		with open('patents.msgpack', 'wb') as f:
			print 'still dumping'
			msgpack.pack(d, f)
		print 'done dumping'
	return patents

# Create network category dictionary: {main_uspto: index of category in adj matrix}
def create_network_category_dict(patents):
	# First find out how many distinct categories there are
	categories = set() # Set containing all main_uspto values
	cat_dict = {} # Dictionary mapping each main_uspto to its index value in adjacency matrix

	for d in patents.itervalues():
		categories.add(d['main_uspto'])

	counter = 0
	for item in categories:
		cat_dict[item] = counter
		counter += 1

	return cat_dict

# Create vector and adjacency matrix ###### CREATE A LIST OF ADJACENCY MATRICES SPANNING n YEARS EACH
# fn = filename of csv, patents = dictionary of patents, fyear_gap = maximum allowed years between cited and citing patent
def create_vector_and_matrix(patents, start_year, end_year, fyear_gap):
	cat_dict = create_network_category_dict(patents) # Dictionary mapping each main_uspto to its index value in adjacency matrix
	n = len(cat_dict)

	# Initialize the list of vectors and matrices
	num_years = end_year - start_year
	vectors = [np.zeros((n, 1)) for i in range(num_years)] # Create an Nx1 vector
	matrices = [np.zeros((n, n)) for i in range(num_years)] # Create an NxN adjacency matrix

	# Find all csv files in cit_received directory
	files = [file for file in os.listdir('./data/cit_received') if file.endswith(".csv")]

	counter1 = 0
	# Read in each file in directory
	for f in files:
		print './data/cit_received/' + f
		# Read csv file in chunks (takes too much memory to load all at once)
		# Only the first csv file has headers
		if f == 'patents_cit_received_part1.csv':
			reader = pd.read_csv('./data/cit_received/' + f, chunksize=50)
		else:
			reader = pd.read_csv('./data/cit_received/' + f, chunksize=50, header=None)
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
				fyear, cat = patents[patnum]['fyear'], patents[patnum]['main_uspto']
				# If fyear is NaN, just skip this entry
				if pd.isnull(fyear):
					continue
				else:
					fyear = int(fyear)
				i = fyear - start_year # index of this year's matrix inside the adj_matrices list
				j = cat_dict[cat] # index in adjacency matrix for current patent's category
				# Iterate through each other patent that cites the current patnum
				for cit_by in row:
					# Move on to next row if cit_by is NaN
					if pd.isnull(cit_by):
						break
					else:
						cit_fyear, cit_cat = patents[cit_by]['fyear'], patents[cit_by]['main_uspto']
						# If the fyear for this citing patent is null, skip it
						if pd.isnull(cit_fyear):
							continue
						# If citing patent is filed more than fyear_gap years after the cited patent, ignore
						elif cit_fyear - fyear  - 1 > fyear_gap or cit_fyear - fyear <= 0:
							continue
						else:
							# Add entry to vector
							vectors[i][j] += 1
							# Add entry into adjacency matrix
							k = cat_dict[cit_cat] # adj matrix index for the category of patent that cited this patent
							if j == k:
								matrices[i][j,k] += 1 # add 1 to the (j,k) element of the ith matrix
							else:
								matrices[i][j,k] += 1
								matrices[i][k,j] += 1
	# Save vectors and matrices into serialized files
	print 'dumping vectors'
	with open('vectors.msgpack', 'wb') as f:
		msgpack.pack(vectors, f)
	print 'dumping matrices'
	with open('matrices.msgpack', 'wb') as f:
		msgpack.pack(matrices, f)
	print 'dumping cat_dict'
	with open('category_dictionary.msgpack', 'wb') as f:
		msgpack.pack(cat_dict, f)
	print 'done dumping'

	print vectors, matrices

# Define variables
patents = retrieve_patent_data() # Dictionary of format: {patnum: {fyear: int, main_uspto: int}}

create_vector_and_matrix(patents, start_year, end_year, year_gap)

