import pandas as pd
import numpy as np
import random
import string
import metapy
import time
import subprocess
import os
import re
import nltk
import argparse
from nltk.corpus import stopwords
from collections import Counter


NIPS_FPATH = 'papers.csv'
NON_WORDS = re.compile('\W ')
ALPHA_SPACE = r'[^a-zA-Z ]+'
CLEANED_DATA = 'research_papers/research_papers.dat'
CONFIG = 'config.toml'
DATASET_STOPWORDS = 'dataset_stopwords.txt'

SUBSET = 100
MIN_WORD_LEN = 3



def create_document_collection(papers_fname):
	""" 
	Read in csv file of papers and returns dataframe object with CSV fields 
	"""
	df = pd.read_csv(papers_fname)
	df = df[['year', 'title', 'paper_text']] 
	return df

def clean_data(papers):
	""" 
	The NIPS papers given are extremely dirty. This is an attempt to clean them. 
	"""

	cleaned_papers = []

	# Load saved cleaned data if exists
	if os.path.isfile(CLEANED_DATA):
		print('Found cleaned data! Loading from %s' % os.path.abspath(CLEANED_DATA))
		with open(CLEANED_DATA, 'r') as f:
			for line in f:
				cleaned_papers.append(line.split(' '))
		return cleaned_papers

	# Clean our dataset
	with open(CLEANED_DATA, 'a') as f:
		for idx, paper in enumerate(papers):
			print('Cleaning and saving paper %d of %d...' % (idx, len(papers)))

			# Remove stopwords
			stop_words = set(stopwords.words('english'))

			# Tokenize paper
			words = nltk.word_tokenize(paper)

			# Lowercase words
			lower = [word.lower() for word in words]

			# Filter out short, likely garbage words
			clean = [word for word in lower if (len(word) >= MIN_WORD_LEN) and (not word in stop_words)]

			# Remove all words with pure punctuation (more garbage)
			clean = [word for word in clean if not all(char in string.punctuation for char in word)]

			# Remove all words that are not alphabetic
			clean = [word for word in clean if word.isalpha()]

			# Stem words for robustness
			stemmer = nltk.stem.porter.PorterStemmer()
			stemmed = [stemmer.stem(word) for word in clean]
			cleaned_papers.append(stemmed)

			# Save cleaned paper
			joined = ' '.join(stemmed)
			f.write(joined)
			f.write('\n')

	return cleaned_papers

def scale_scores(scores, years, citations, y_w, c_w):
	""" 
	Weight highly cited and more recent papers higher 
	"""
	BASELINE = 1987	# Beginning of NIPS Conference

	offsets = [years[doc[0]] - BASELINE for doc in scores]	# Compute offset year from base year of the top papers

	# Sigmoidal transform
	def year_transform(x, a=10, b=1.5, c=4, d=1):
		transform = c / (1 + np.exp(-a*(x-b))) + d
		return transform

	y_scales = [year_transform(offset) for offset in offsets]

	# Scale the BM25 scores by recency and citation count
	scaled_scores = []
	for idx, score in enumerate(scores):
		scaled_score = score[1] * y_scales[idx]
		scaled_scores.append((score[0], scaled_score))

	return scaled_scores


def rank_documents(query, idx, years, citations, n=10, k=1.2, b=0.75, y=0.1, c=1):
	""" 
	Rank documents according the ranking function score 
	"""
	print('Finding relevant documents to the search term: "%s"' % query)

	# Stem the query before passing into the score for standardization
	stemmer = nltk.stem.porter.PorterStemmer()
	query_words = query.split()
	stem_queries = [stemmer.stem(word) for word in query_words]
	stemmed_query = ' '.join(stem_queries)

	# Compute score for each document
	query_doc = metapy.index.Document()
	query_doc.content(stemmed_query)

	# ranker = ModifiedBM25Ranker(k1=k, b=b)
	ranker = metapy.index.OkapiBM25(k1=k, b=b)
	scores = ranker.score(idx, query_doc, num_results=n)
	scaled_scores = scale_scores(scores, years, citations, y, c)

	# Sort scores
	ranked_scores = sorted(scaled_scores, key=lambda x: x[1], reverse=True)

	return ranked_scores

def find_bigrams(papers):
	""" 
	Find most common bigrams in the papers 
	"""
	print('Finding bigrams...')
	tf_bigrams = []	# Term frequency count of bigrams
	df_bigrams = []	# Document frequency count of bigrams
	for paper in papers:
		# Clean the data, similar to clean_data()
		# Tokenize paper
		words = nltk.word_tokenize(paper)
		# Lowercase words
		lower = [word for word in words]
		# Filter out short, likely garbage words
		clean = [word for word in lower if len(word) >= MIN_WORD_LEN]
		# Remove all words with pure punctuation (more garbage)
		clean = [word for word in clean if not all(char in string.punctuation for char in word)]

		# Store bigrams
		bigrams = list(nltk.bigrams(clean))
		unique_bigrams = set(bigrams)

		tf_bigrams.extend(bigrams)
		df_bigrams.extend(unique_bigrams)

	return tf_bigrams, df_bigrams

def find_datasets(tf_bigrams, df_bigrams, stopwords_fname, beta=5):
	""" 
	Find datasets from bigrams 
	"""
	# Load stopwords for dataset
	dataset_stopwords = set()
	try:
		with open(stopwords_fname, 'r') as f:
			dataset_stopwords = set([word.strip() for word in f.readlines()])
		print('Loaded dataset stopwords from %s!' % stopwords_fname)
	except:
		print('Could not load stopwords from %s, using none.' % stopwords_fname)

	# Find datasets by matching second word to "dataset"
	stop_words = set(stopwords.words('english'))
	tf_dataset = [bigram[0] for bigram in tf_bigrams if (bigram[1] == 'dataset') and (not bigram[0].lower() in dataset_stopwords | stop_words)]
	df_dataset = [bigram[0] for bigram in df_bigrams if (bigram[1] == 'dataset') and (not bigram[0].lower() in dataset_stopwords | stop_words)]

	tf_counter = Counter(tf_dataset)
	df_counter = Counter(df_dataset)

	# Now rank the datasets using a geometric F-score (Increasing "beta" places more weight on DF, decreasing it places mores weight on TF)
	scores = []
	for dataset in tf_counter:
		score = (1 + beta**2) * tf_counter[dataset] * df_counter[dataset] / (beta**2 * tf_counter[dataset] + df_counter[dataset])
		# Weird heuristic, but datasets tend to be acronyms/capitalized, weigh these more heavily
		num_capital = sum(1 for letter in dataset if letter.isupper())
		# Check if there's more than one letter capitalized in the dataset
		if num_capital >= 1:
			score *= 5

		scores.append((dataset, score))

	sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)

	return sorted_scores

def find_trigrams(papers):
	""" 
	Find most common bigrams in the papers 
	"""
	print('Finding trigrams...')
	tf_trigrams = []	# Term frequency count of bigrams
	df_trigrams = []	# Document frequency count of bigrams
	for paper in papers:
		# Clean the data, similar to clean_data()
		# Tokenize paper
		words = nltk.word_tokenize(paper)
		# Lowercase words
		lower = [word for word in words]
		# Filter out short, likely garbage words
		clean = [word for word in lower if len(word) >= MIN_WORD_LEN]
		# Remove all words with pure punctuation (more garbage)
		clean = [word for word in clean if not all(char in string.punctuation for char in word)]

		# Store bigrams
		trigrams = list(nltk.trigrams(clean))
		unique_trigrams = set(trigrams)

		tf_trigrams.extend(trigrams)
		df_trigrams.extend(unique_trigrams)

	return tf_trigrams, df_trigrams

def find_methods(tf_trigrams, df_trigrams, beta=1):
	""" 
	Find methods from trigrams 
	"""
	# Find methods by matching the word "use"
	stop_words = set(stopwords.words('english'))
	tf_method = [(trigram[1], trigram[2]) for trigram in tf_trigrams if (trigram[0] == 'use') and (not trigram[1].lower() in stop_words) and (not trigram[2].lower() in stop_words)]
	df_method = [(trigram[1], trigram[2]) for trigram in df_trigrams if (trigram[0] == 'use') and (not trigram[1].lower() in stop_words) and (not trigram[2].lower() in stop_words)]

	tf_counter = Counter(tf_method)
	df_counter = Counter(df_method)

	# Now rank the methods using a geometric F-score (Increasing "beta" places more weight on DF, decreasing it places mores weight on TF)
	scores = []
	for method in tf_counter:
		score = (1 + beta**2) * tf_counter[method] * df_counter[method] / (beta**2 * tf_counter[method] + df_counter[method])
		# Slightly less relevant than datasets, but methods tend to be acronyms/capitalized, weigh these more heavily
		num_capital = sum(1 for word in method for letter in word if letter.isupper())
		# Check if there's more than one letter capitalized in the method
		if num_capital >= 1:
			score *= 2

		scores.append((method, score))

	sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)

	return sorted_scores


def main(query):

	# Gather data
	collection = create_document_collection(NIPS_FPATH)

	# Clean data
	titles = collection.title
	texts = collection.paper_text
	years = collection.year

	processed_papers = clean_data(texts)

	# Make our indices
	print('Making inverted index based on dataset...')
	inv_idx = metapy.index.make_inverted_index(CONFIG)
	
	# Form query and rank documents based on it
	citations = None
	scores = rank_documents(query, inv_idx, years, citations, n=100)

	# Retrieve top documents
	top_ids = [doc_id[0] for doc_id in scores]
	results = collection.iloc[top_ids][['title', 'year']]

	# Retrieve text features
	top_papers = texts[top_ids]
	tf_bigrams, df_bigrams = find_bigrams(top_papers)
	tf_trigrams, df_trigrams = find_trigrams(top_papers)

	# Find information of interest
	datasets = find_datasets(tf_bigrams, df_bigrams, DATASET_STOPWORDS)	
	methods = find_methods(tf_trigrams, df_trigrams)

	# Return information
	ret_titles = results[:10].title.tolist()
	ret_years = results[:10].year.tolist()
	ret_datasets = [dataset[0] + ' dataset' for dataset in datasets[:5]]
	ret_methods = [' '.join(method[0]) for method in methods[:5]]

	print('Papers:')
	for idx, paper in enumerate(ret_titles):
		print('Year: %s\tTitle: %s' % (ret_years[idx], ret_titles[idx]))
	
	print('Datasets: ' + str(ret_datasets))
	print('Methods: ' + str(ret_methods))

	return ret_titles, ret_years, ret_datasets, ret_methods

	
if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("-q", "--query", help="Enter your query here", required=True)
	args = parser.parse_args()
	main(args.query)
