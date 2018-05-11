
# EZ Review

{ Insert intro here}

## What does this do?

{ Insert here }

## How do I get this working?

This project uses Python 3 , so make sure you have that before continuing. Specifically, I run it on Python 3.6.

 1. First clone the repository to your local machine.
 2. You will also need the data which can be found [here](https://www.kaggle.com/benhamner/nips-papers/data). Please download `papers.csv`. You will need to unzip the download and place the `papers.csv` in the **top-level directory** of the cloned repository.
 3. Now you can install the necessary dependencies, like below:
```bash
pip3 install -r requirements.txt 
```

You can finally run the code now! Let's start off by using the CLI, and let's do a search on the query `text mining`.
```bash
python3 ez_review.py -q "text mining"
```
**NOTE:** This will take a *long time* on the first run. The original `papers.csv` file is dirty, and thus we need to clean the data at least once. Don't worry! The cleaned data is saved to disk (approximately 100 MB) after it is allowed to run once. This will also create an index in a directory `./idx/` on your computer so that the application can perform faster searches

**NOTE:** Do NOT terminate the script while it is cleaning. If you do, you will need to delete the incomplete `./research_papers/research_papers.dat` file if it was created. Otherwise, the program will think the data has already been cleaned!

## Interpreting the Results
After 10 to 20 seconds, your results should appear, resulting in a list of relevant papers that you may want to read relating to that topic query! In addition, you will see a list of possible datasets you may want to look further into for your research, as well as popular methods used in the area! Here is the sample output when we input `text mining`:
```
Papers:

Year: 2016  Title: Anchor-Free Correlated Topic Modeling: Identifiability and Algorithm

Year: 2005  Title: Sequence and Tree Kernels with Statistical Feature Mining

Year: 2007  Title: Mining Internet-Scale Software Repositories

Year: 2010  Title: Deterministic Single-Pass Algorithm for LDA

Year: 2009  Title: Dirichlet-Bernoulli Alignment: A Generative Model for Multi-Class Multi-Label Multi-Instance Corpora

Year: 2001  Title: Hyperbolic Self-Organizing Maps for Semantic Navigation

Year: 2010  Title: b-Bit Minwise Hashing for Estimating Three-Way Similarities

Year: 2004  Title: On Semi-Supervised Classification

Year: 2001  Title: A kernel method for multi-labelled classification

Year: 2004  Title: Conditional Models of Identity Uncertainty with Application to Noun Coreference

Datasets: ['Reuters-21578 dataset', 'WebKB dataset', 'Reuters dataset', 'Newsgroups dataset', 'MNIST dataset']

Methods: ['LDA discover', 'Metropolis-Hastings sampler', 'dictionary definitions', 'second-order Taylor', 'Minkowski space']
```

## Using the GUI 

If you aren't a fan on using a CLI, there is also an associated front-end in order to use this application! Unfortunately, there is no public server to visit the website, so you will still need to launch it from the terminal.

```bash
python3 app.py
```

This will launch a Flask server on your local machine which will be listening for requests on port 8080. After you see the following message:

`* Running on http://0.0.0.0:8080/ (Press CTRL+C to quit)`

You can then visit http://localhost:8080/ on your favorite browser and interact with the search engine in a traditional fashion! Again, it will take around 20 seconds to retrieve your results before it loads onto the webpage.

