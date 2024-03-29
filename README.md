[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4680673.svg)](https://doi.org/10.5281/zenodo.4680673)

# DoConA: Document Content and Citation Analysis Tool

DoConA is an open source, configurable, extensible Python tool to analyse the level of agreement between the citation network of a generic set of textual documents and the textual similarity between these documents.

## Running the pipeline

Please read the following instructions carefully before running the pipeline. If you encounter problems or have feature suggestions, please file an [issue](https://github.com/MaastrichtU-IDS/docona/issues)

##### Software requirements:

+ [Python 3.7+](https://www.python.org/downloads/)
+ [Git](https://git-scm.com/) **or** an archive extractor like [7-Zip](https://www.7-zip.org/)

##### Data requirements:

+ A set of input text documents in `.txt` format. The filename for each document should be a unique identifier for that document. E.g. `62016CJ0295.txt`
+ A CSV file named `citations.csv` representing the citation network of the input text documents. The file should have exactly two columns with headers `source` and `target` respectively in that order. The column data should consist exclusively of document identifiers (without the `.txt` file extension for the document. E.g. `62016CJ0295`). `source` identifiers represent the citing documents and `target` identifiers represent the cited documents
+ A CSV file called `sample.csv` which contains a representative sample of the document identifiers in your input corpus. The file should contain exactly one column with no header and each document identifier should appear on a new line. The pipeline will be executed on the sample of documents specified in this file
+ **Optional:** A CSV file called `stopwords.csv` which contains a list of words which should be removed from each text document during the preprocessing phase of the DoConA pipeline. The file should contain exactly one column with no header and each word should appear on a new line in the file 
+ **Optional:** one or more pretrained word2vec embedding files 

##### Step 1: get a copy of the repository
    
+ If you are using Git, type this command in your commandline tool: `git clone https://github.com/MaastrichtU-IDS/docona.git`
+ If you are using an archive extractor, download this repository's code archive from [here](https://github.com/MaastrichtU-IDS/docona/archive/master.zip) and extract it to the desired folder on your file system

##### Step 2: provide your data (documents, citation network and sample)
    
+ Place your input text documents in the folder `inputdata/fulltexts/` 
+ Place your `citations.csv` and `sample.csv` files in the folder `inputdata/`
+ **Optional:** if you wish to provide [stop words](https://nlp.stanford.edu/IR-book/html/htmledition/dropping-common-terms-stop-words-1.html), place your custom `stopwords.csv` file in the folder `inputdata/resources/`
+ **Optional:** if you wish to provide custom pretrained word embeddings to use with the pipeline, place these in the folder `inputdata/resources/`
+ If you have chosen to provide the optional pretrained embeddings, you will have to study and modify the script `docona/docona.py` to include the following code snippets (copy and paste these into that script) where it states:

	`# --- ADD CUSTOM PRETRAINED MODEL CODE HERE --- #`

You have to paste the following lines of code (replace the relevant parts where necessary):

	from pretrainedmodels import getdoc2vecmodel,getword2vecmodel
	desired_name_of_model_1 = getdoc2vecmodel("name_of_input_pretrained_embedding_file.extension", "desired_name_of_output_adapted_model.extension")									
	desired_name_of_model_2 = getword2vecmodel("name_of_input_pretrained_embedding_file.extension", "desired_name_of_output_similarity_matrix_file", binaryfile="True/False")

	from pretrainedmodelssimilarity import dosimilaritychecks
	dosimilaritychecks("doc2vec", "desired_name_for_input_embeddings_1", desired_name_of_model_1, "cosine")
	dosimilaritychecks("word2vec", "desired_name_for_input_embeddings_2", desired_name_of_model_2, "wmd")`

+ **NB:** For the `binaryfile` variable, `"True/False"` should be replaced with `True` if the embedding file is a binary file and `False` otherwise. This code adapts and retrains your custom embeddings to use with the pipeline.
+ **NNB:** `desired_name_for_input_embeddings_1` and `desired_name_for_input_embeddings_2` are different names that will appear in the output of the pipeline and are **not** the filename of the input embeddings. They will be used in the generated report to associate the results of the pipeline with the different models. You can choose any name for these strings such that you will easily recognise the different models in the output results of the pipeline

##### Step 3: run the pipeline

+ In your commandline tool, change into the root directory of your local copy of the repository
+ **Important:** create a fresh [virtual environment](https://docs.python.org/3/tutorial/venv.html) with a default or base installation of Python 3.7+  
+ Run the command `pip install -r requirements.txt` which will install all required libraries for DoConA in your fresh Python virtual environment
+ On installation completion of the required libraries in the previous step, change into the `docona/` directory of your local copy of the code (this directory contains all `.py` code files for the tool)
+ Run the command `python docona.py`. **Note:** the pipeline can take a while to run on a mid-range machine (E.g. Quad-core, 16GB of RAM). For example, given 10,000 documents and a representative random sample of 500 documents, with two additional pretrained models, the pipeline can take close to 24 hours to run on such a machine. It is recommended to run the pipeline on a high performance computing platform
+ On successful completion of the pipeline, there will a `results.csv` placed in the folder `outputdata/`. DoConA will also generate various other model and data files during the run and will place these in the folders `inputdata/resources/` and `outputdata/`
+ Please see the [wiki](https://github.com/MaastrichtU-IDS/docona/wiki) for more detailed information about the generated data

## License
Copyright (C) 2020, Kody Moodley and Pedro Hernandez Serrano

This program is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License as published by the Free Software Foundation, either version 3 of the License, or any later version.
