# CourseProject script setup instruction for Reproducing paper "Generating Semantic Annotations for Frequent Patterns with Context Analysis" 

1. Overview of the github CourseProject 

	There are two folders, Datasets and PythonCodes, in the repository. All python scripts used to reproduce the paper using the DBLP dataset (paper section 5.1) are located in the PythonCodes folder. All dataset imported and generated using the python scripts are located int he Datasets. The input and output paths of these Datasets files need to be changed if downloaded to your local computer.

	In addition, the final report and the powerpoint slides of demo guidance are also in the repository. 
	

2. Script running instruction
	2.1. Parsing of raw data
		The DBLP dataset was downloaded from "https://hpi.de/naumann/projects/repeatability/datasets/dblp-dataset.html" as xml file. This raw dataset is "dblp50000.xml". 
		Parse "dblp50000.xml" by script "DBLP_raw_data_parsing.py" and generate the dataset "DBLP2000.csv". 

	2.2. Building the Context Units Space (finding context indicators)

	2.2.1. Author closed Frequent Pattern (FP) mining using FPgrowth algorithm in MLXtend Lib. 
		Process the author column of "DBLP2000.csv" by script "Author_FP_mining.py" and generate the dataset "authorsFP2000_with_index.csv". This output dataset contains 14 closed FPs of authors and each author FP has its list of transaction index of "DBLP2000.csv" (e.g. author "Edwin R. Hancock" is a closed FP and it showed in the 839th, 1119th, 1127th, 1204th and 1576th row of DBLP2000, its transaction index list is [839, 1119, 1127, 1204, 1576] ).  
	
	2.2.2. DBLP title preprocessing
		Process the title column of "DBLP2000.csv" by script "DBPL_preprocessing_titles.py" and generate the dataset "DBLP2000_preprocessed_titles.txt". In this step, stop words are removed and the titles are stemmed. 

	2.2.3. Title sequential Pattern mining using PrefixSpan algorithm in PySpark
		Process the "DBLP2000_preprocessed_titles.txt" using "titles_seqPattern_mining.py" to find closed sequential frequent patterns from titles of DBLP2000. This generate a output folder containing the pattern file "part-00000". Set the "part-00000" file to txt format. 
		Process "part-00000.txt" using "Title_sequentialFP_processing.py" to generate the cleaned dataset "titlesFP2000.csv".

	2.2.4. Find transaction index of title sequential patterns
		Process "titlesFP2000.csv" using "Find_transaction_index_of_title_FPs.py" to generate "titlesFP2000_with_index.csv". List of transaction index is associated with each title pattern.

	2.2.5. Title redundancy reduction by microclustering (hierarchical clustering)
		To furhter reduce the redundancy of the title patterns, process "titlesFP2000_with_index.csv" using script "Hierarchical_clustering_titleFPs2000.py". This script applys hierarchical clustering with Jaccard Distance defined per paper and clusters 1912 title patterns into 166 clusters and choose the most frequent pattern in each cluster as the "centroid" pattern to further build the contex units space.

	2.2.6. Combine author FPs and title FPs to build the context units space
		Combine 'authorsFP2000_with_index.csv' and 'titlesFP2000_final.csv' using script "DBLP2000_context_units_with_transaction_index.py" to generate the final context units dataset "DBLP2000_context_units.csv".

	2.3. Annotate given frequent patterns using context modeling 

	2.3.1. Build weight vectors of FPs in the context units space

		We use all closed frequent patterns of authors and titles themselves as patterns and build weight vectors for them in the context units formed by themselves. Process ""DBPL2000_context_units.csv" and "DBLP2000.csv" using script "Weighting_function.py" to generate "Context_units_weights.csv". This is a weight matrix between pairwised patterns (FP as given pattern and FP as context unit). Each element of the matrix is the Mutual Information between the patterns per definition in the paper. 

	2.3.2. Annotate the given FP (e.g. an author) by context units with highest weights

		Process "Context_units_weights.csv" using script "Annotating_pattern.py". We pick the first author and rank the weights of its context vector. The context units with highest 10 weights are selected as the annotation of this author. The output dataset is "author_annotation_example1.csv".

	2.4. Find representative titles of the given author

		Process "DBLP2000_context_units.csv", "DBLP2000.csv" and "Context_units_weights.csv" using script "Find_representative_titles_to_author.py". This script first generates the weight matrix of transactions (titles) in the context units space as the dataset "transaction_weights.csv", and then compute the cosine similarity between the transaction weight vectors and the given author weight vector (e.g. the same author chosen in 2.3.2) and generate the top 10 representative titles with highest similarity scores as dataset "rep_titles_example1.csv".

	2.5. Find synonyms of the given author

		Process "Context_units_weights.csv" using script "Find_synonyms.py" to compute the cosine similarity between the candidate pattern weight vectors (e.g. all closed frequent patterns of authors) and the given author weight vector (e.g. the same author chosen in 2.3.2). Select the authors with highest r similarity scores as the synonyms of the given author other than the author himself (the author is always the most similar pattern to himself). This gives output dataset "synonyms_to_author_example1.csv".

	2.6. A final display of the context annotation of the given author

		Combine ""author_annotation_example1.csv", "rep_titles_example1.csv" and "synonyms_to_author_example1.csv" using script "Author_context_annotation_example1.py" to generate the dataset "context_annotation_example1.csv" that shows all three parts of the context annotation of a given author in a DBPL dataset: annotation of the author, representative publication (titles) of the author and the synonyms (co-authors) of the author. 