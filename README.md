
# EZ Review

Created by Shandilya (Shawn) Pachgade. Originally created for CS 410 (Text Information Systems) at the University of Illinois at Urbana-Champaign.

## What does this do?

This is a search engine aimed at researchers who are in the beginnings of a research project, specifically in their literature review stage. This application retrieves relevant research papers that a researcher might find interesting to read. This is similar to Google Scholar, but in addition, it finds popular datasets and methods that are used in the field! A major point about this project is that can be used in any field! The only requirements is that you need a collection of research papers. For this short introduction, we will be applying this to the field of data science and machine learning related fields!

## How does it work?

These are some technical details of this project that you might find interesting, feel free to skip to the next section if you just want to run the code.

### Cleaning the Data
I primarily relied on information retrieval techniques when creating this application. First, I went through every single research paper in the dataset, and cleaned the text up. This entailed removing garbage characters, unlikely sequences of characters, stopwords, and then finally stemming each of the surviving words (I used the Porter stemmer).

### Ranking the Papers
Once the data has been cleaned, we can finally start ranking! For a lot of this, I used the [Python bindings for MeTA](https://github.com/meta-toolkit/metapy) to help. I created an inverted index, which maps a list of terms to documents that contain these terms, which helps speed up the processing time of a search query. Given the user's query input, we need to clean it so it matches with the terms available in the inverted index, thus we tokenize the query and then stem it. 

To actually rank the documents, we use [Okapi BM25](https://en.wikipedia.org/wiki/Okapi_BM25), a popular ranking function that uses TF-IDF to favor documents with higher counts of the query terms, while also penalizing document length and minimizing the effect of frequently occurring query terms.  

However, we also want to incorporate some available metadata we have about our document collection in the ranking function. It is likely that more recent papers are likely to have better performance and more novelty than those that are older, and thus be more useful and relevant. It is important to not weigh recent papers too heavily, so I scale the BM25 score by a sigmoidal function, which rapidly picks up more scaling towards the middle. Specifically, the inflection point is picked at the halfway mark between the year that the NIPS conference began (we will be testing on NIPS papers), and the current year.  

The top-scoring research papers are then returned to the user as relevant documents.

### Finding Datasets
First a list of bigrams is generated from the list of relevant documents, and then matched the second term to the word "dataset". Aggregating all of these bigrams, we found a list of candidate datasets that might be of use, however, some of these might not actually be datasets. In an effort to combat this issue, I created a separate stopwords list used specifically when searching for datasets. This list can be found (and modified) at `dataset_stopwords.txt`. This stopword list was created by aggregating the bigrams to create a background language model for datasets, and then manually picked from the list.

A set of heuristics then come into play, one of which is a combined term-frequency and document-frequency scheme. We favor datasets that appear in multiple documents rather than just one (DF transform), however we also care about how often a dataset appears overall (TF transform) as a measure to verify that the dataset actually exists, and isn't just a stopword or meaningless. To weigh document-frequency over term-frequency, a geometric mean balancing the two is employed, specifically a F-measure with $\beta=5$. In addition, most datasets tend to be be acronyms, and therefore capitalized, thus we weight these candidate datasets more as well. The top scoring candidates are returned as relevant datasets of potential interest.

### Find Methods
This is very similar to finding datasets, except instead of finding bigrams, we find trigrams, and match the first term to the word "use" to find candidates. The concept is the same however, except with different parameter values, documented in the code. One of these is that we use $\beta=1$ when computing the F-measure. To enhance performance, other matches can be used, as we are likely ignoring other methods that never become candidates. 

## Show me how to run it!

This project uses Python 3, so make sure you have that before continuing. Specifically, I run it on Python 3.6.

 1. First clone the repository to your local machine.
 2. You will also need a collection of research papers so that we can actually rank documents! We will use the NIPS dataset found [here](https://www.kaggle.com/benhamner/nips-papers/data). Please download `papers.csv`, unzip the download, and place the `papers.csv` in the **top-level directory** of the cloned repository.
 3. Now you can install the necessary dependencies, like below:
```bash
pip3 install -r requirements.txt 
```

You can finally run the code now! Let's start off by using the CLI, and let's do a search on the query `text mining`.
```bash
python3 ez_review.py -q "text mining"
```
**NOTE:** This will take a *long time* on the first run. The original `papers.csv` file is dirty, and thus we need to clean the data at least once. Don't worry! The cleaned data is saved to disk (approximately 100 MB) after it is allowed to run once. This will also create an index in a directory `./idx/` on your computer so that the application can perform faster searches.

**NOTE:** Do NOT terminate the script while it is cleaning. If you do, you will need to delete the incomplete `./research_papers/research_papers.dat` file if it was created. Otherwise, the program will think the data has already been cleaned!

## Interpreting the Results
After 10 to 20 seconds, your results should appear, resulting in a list of relevant papers that you may want to read relating to that topic query! In addition, you will see a list of possible datasets you may want to look further into for your research, as well as popular methods used in the area! Here is the sample output when we input `text mining`:
```
Year: 2016  Title: Anchor-Free Correlated Topic Modeling: Identifiability and Algorithm

Year: 2007  Title: Mining Internet-Scale Software Repositories

Year: 2010  Title: Deterministic Single-Pass Algorithm for LDA

Year: 2005  Title: Sequence and Tree Kernels with Statistical Feature Mining

Year: 2009  Title: Dirichlet-Bernoulli Alignment: A Generative Model for Multi-Class Multi-Label Multi-Instance Corpora

Year: 2010  Title: b-Bit Minwise Hashing for Estimating Three-Way Similarities

Year: 2014  Title: Content-based recommendations with Poisson factorization

Year: 2009  Title: Learning Bregman Distance Functions and Its Application for Semi-Supervised Clustering

Year: 2010  Title: Latent Variable Models for Predicting File Dependencies in Large-Scale Software Development

Year: 2016  Title: DeepMath - Deep Sequence Models for Premise Selection

Datasets: ['Reuters-21578 dataset', 'WebKB dataset', 'Reuters dataset', 'Newsgroups dataset', 'MNIST dataset']

Methods: ['LDA discover', 'Metropolis-Hastings sampler', 'dictionary definitions', 'second-order Taylor', 'CTPF recommend']
```

## Using the GUI 

If you aren't a fan on using a CLI, there is also an associated front-end in order to use this application! Unfortunately, there is no public server to visit the website, so you will still need to launch it from the terminal.

```bash
python3 app.py
```

This will launch a Flask server on your local machine which will be listening for requests on port 8080. After you see the following message:

`* Running on http://0.0.0.0:8080/ (Press CTRL+C to quit)`

You can then visit http://localhost:8080/ on your favorite browser and interact with the search engine in a traditional fashion! Again, it will take around 20 seconds to retrieve your results before it loads onto the webpage.

## Future Work

### Citation Statistics

In the future, I would really like for this project to incorporate citation data into the ranking system. Unfortunately, I could not find any readily available citation graph which would be easy to integrate into this project. [Microsoft Cognitive Services](https://azure.microsoft.com/en-us/services/cognitive-services/) offers an Academic Graph API, although I have not been successful in using it and retrieving results.

Once we have access to citation data, then weighing papers with higher citation counts, or having more sink-like behavior in the graph, should be scored as more relevant. It might be useful to jointly optimize the weighting of citations and year together, as the older a paper is, the more likely it is to be cited. The magnitude of how much to emphasize citation statistics is yet to be determined, however it will likely be heuristic. Unless...

### Automatic Learning of Parameters

Optimization of these weight parameters might be possible through some sort of machine learning algorithm, given that there is a labelled dataset. This labelled dataset will likely have to be collected in an online setting in which explicit feedback is given on whether an individual document is relevant. 

After obtaining a labelled dataset, we could treat the problem as a binary classification problem of relevant versus non-relevant. Then, ranking can be done in by taking the documents furthest away from the decision boundary. The problem with this is the binary classification problem exists for every single possible set of queries (infinite possibilities!), therefore the optimization problem is in dire need of reformulation if this approach is to be used.

