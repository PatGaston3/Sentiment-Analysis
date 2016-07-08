# Sentiment-Analysis
(Natural Language Processing Project using NLTK)

The goal of this project is to implement a system that automatically extracts sentiment from an input. I used NLTK
Data as a knowledge base for the program, extracting tweets as data to train the machine learning algorithm (Naive Bayes).

The term is over (and I have graduated, yesss) but I would like to further develop my program as a learning/fun experience.

Currently, I would like to focus on including bigrams in my algorithm for more precision *as well* add a 'neutral' sentiment for
sentences like "The house is yellow".


Required Libraries
------------------

This project uses several libraries that either need to be installed or
need to be present in the project's `lib/` directory. The following is a
list of the required libraries, as well as at least one way (source) to
obtain the library.

### NLTK

Natural Language Processing (NLP) functions such as sentence
segmentation, word tokenization, and more.

### Installing NLTK
See [NLTK installation guide](http://nltk.org/install.html)

First download `setuptools`, http://pypi.python.org/pypi/setuptools
```
sudo sh Downloads/setuptools-...egg
sudo easy_install pip 
sudo pip install -U numpy
sudo pip install -U pyyaml nltk
```

#### nltk resources

In addition, you will need to download several nltk resources using
nltk.download() after you have the nltk library installed.

### Download NLTK datasets

####Important!! 
make sure that you have the training data from NLTK here at: http://www.nltk.org/nltk_data/
the 'twitter_samples' file is #74 in the list. Click link for direct download.

```
python
>>> import nltk
>>> nltk.download()
```

Once the NLTK Downloader GUI pops up, download all to `/Users/USERNAME/nltk_data`


#####Note*

I have only ran this program on a MAC OS and these setup instructions are for mac users. If you aren't a mac user and the instructions differ,
please email me so I can update the ReadMe.


Patrick Gaston

patrick.e.gaston@gmail.com
