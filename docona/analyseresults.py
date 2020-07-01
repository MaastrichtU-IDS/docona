#!/usr/bin/env python
# # Document Content and Citation Analysis (DoConA)
# # Module: analyses "results.csv" raw output from pipeline and computes the similarity overlap between the citation network of the documents and the document texts 
# # Output: generates "analysis.csv" file with summary of the results
# GNU AGPLv3 - https://choosealicense.com/licenses/agpl-3.0/

import pandas as pd
import os

def analyse():
	# Import results and remove duplicate rows
	results_df = pd.read_csv(os.path.join(os.path.join(os.path.realpath('..'), "outputdata"), "results.csv"))
	results_df.drop_duplicates(subset=None, keep='first', inplace=True)

	# Filter for the overlaps and sort the dataframe (in descending order) according to the number of citation overlaps (group by method as well)
	overlaps_df = results_df[results_df["citation_link"] == True]
	overlaps_df = overlaps_df.groupby(['method']).citation_link.agg('count').to_frame('num_cite_and_similar_links').reset_index()
	overlaps_df = overlaps_df.sort_values('num_cite_and_similar_links',ascending=False)

	# Import full citation network of documents
	citations = pd.read_csv('../inputdata/citations.csv')

	# Function to location the cited documents of a given document
	def find_cited_documents(documentID):
	    relevantsource = citations[citations['source'] == documentID]
	    return relevantsource['target'].tolist()

	# Import sample documents
	unique_samples = pd.read_csv('../inputdata/sample.csv', header=None)

	# Compute the total number of unique documents cited by the sample documents
	cited_documents = []
	for item in unique_samples.values.tolist():
	    cited_documents.extend(find_cited_documents(item[0]))
	num_cite_links = len(cited_documents)

	# Aggregate all the counts into a single data frame and compute the percentage overlaps of each method
	percentage_overlap_arr = []
	num_cite_links_arr = []
	num_rows = len(overlaps_df.index)
	num_cite_and_similar_links_arr = overlaps_df['num_cite_and_similar_links'].values.tolist()

	for i in range(0,num_rows):
	    num_cite_links_arr.append(num_cite_links)
	    current_perc_overlap = (num_cite_and_similar_links_arr[i] / num_cite_links) * 100
	    percentage_overlap_arr.append(current_perc_overlap)

	overlaps_df['num_cite_links'] = num_cite_links_arr
	overlaps_df['percentage_overlap'] = percentage_overlap_arr

	# Write the output to file
	overlaps_df.to_csv('../outputdata/analysis.csv', index=False)

