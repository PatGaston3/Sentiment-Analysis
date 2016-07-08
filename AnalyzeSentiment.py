# sentiment.py
# by: Patrick Gaston
# Twitter Sentiment Analysis
# Computational Linguistics Final Project SPRING 2016

''' Automatic sentiment (positive or negative) extractor from an input '''

# import libs and packages
import nltk
import re
import nltk.classify.util
from nltk.corpus import *
from nltk.corpus import TwitterCorpusReader
from nltk.corpus import twitter_samples
from nltk.classify import NaiveBayesClassifier


## --------------------------------PREPROCESS TRAINING DATA----------------------------------------- ##

'''  
	PREPROCESS Tweets    
		steps:
			1. change '@xxx' to '@user' (@EaglesFan4Life == '@user')
			2. lowercase all characters ('Thanks' == 'thanks')
			3. remove hashtags ('#love' == 'love')
			4. remove multiple white spaces ('       ' == ' ')
			5. remove punctuation ('Hey!!!!!' == 'Hey')
			6. remove URLs ('https:twit.co...' == 'URL')
			7. if more than 3 consonants in sequence, reduce to 2 ('yeaaaaaaaah' == 'yeaah')
				
	'''

def process_tweet(tweets):
	newtweets = []
	for items in tweets:
		newitem = []
		for item in items:
			if len(item) >= 3:
				item = re.sub(r'(.)\1+', r'\1\1', item)
				item = item.lower()
				if '@' in item:
					item = '@user'
				elif '#' in item:
					item = item[1:]
				elif item == 'luv':
					item = 'love'
				elif 'http' in item:
					item = 'UR'
				else:
					pass
		
				newitem.append(item)
		newtweets.append(newitem)
	return newtweets

	
''' create tuples in list of tweets and add sentiment "positive" or "negative" to tweets 

(['@user', 'hey', '!', ':)', 'long', 'time', 'no', 'talk',], 'positive')
(['@user', 'as', 'matt', 'would', 'say', '.', 'welcome', 'to', 'adulthood', '..', ':)', 'URL'], 'positive')

'''	

def get_words(tweets):
	wholelist = []
	for (words, sentence) in tweets:
		wholelist.extend(words)
	return wholelist

def get_word_feats(wordlist):
	wordlist = nltk.FreqDist(wordlist)
	word_feats = wordlist.keys()
	return word_feats
	
''' FEATURE EXTRACTIOIN!

extract features from data sets to implement in machine learning algorithm (Classifier: Naive Bayes)

'''

def extract_feats(doc):
	doc_words = set(doc)
	features = {}
	for word in doc:
		features['contains(%s)' % word] = (word in doc_words)
	return features


def add_pos_sentiment(tweets):
	newlist = []
	for items in tweets:
		sentence, sentiment = items, "positive"
		newlist.append((sentence, sentiment))
	return newlist

def add_neg_sentiment(tweets):
	newlist = []
	for items in tweets:
		sentence, sentiment = items, "negative"
		newlist.append((sentence, sentiment))
	return newlist

## --------------------------------CLASSIFIER + EXECUTION---------------------------------------- ##



def main():
	''' IMPORT DOWNLOADED TWEETS
	steps:
			1. download twitter_samples NLTK corpus for training data
			2. indicate path in directory and assign to variable
			3. Read file to/with TwitterCorpusReader to access methods (Here I need .tokenized())  
	'''
	
	# assign variable to path of training data			
	posroot = '/Users/Pat-Levi-Gaston/nltk_data/corpora/twitter_samples/positive_tweets.json'
	negroot = '/Users/Pat-Levi-Gaston/nltk_data/corpora/twitter_samples/negative_tweets.json'

	pos_reader = TwitterCorpusReader(posroot, '.*\.json')
	neg_reader = TwitterCorpusReader(negroot, '.*\.json')

	raw_pos_tweets = pos_reader.tokenized(".")
	raw_neg_tweets = neg_reader.tokenized(".")
	
	
	pos_tweets = process_tweet(raw_pos_tweets)
	neg_tweets = process_tweet(raw_neg_tweets)

	pos_tweets = add_pos_sentiment(pos_tweets)
	neg_tweets = add_neg_sentiment(neg_tweets)

	tweets = pos_tweets[:2000] + neg_tweets[:2000] # [:3750] original
	testers = pos_tweets[2000:2250] + neg_tweets[2000:2250] # [3750:] original

	word_feats = get_word_feats(get_words(tweets))

	training_set = nltk.classify.apply_features(extract_feats, tweets)
	testing_set = nltk.classify.apply_features(extract_feats, testers)
	
	
	
	classifier = nltk.NaiveBayesClassifier.train(training_set)
	print('train on %d instances, test on %d instances' % (len(tweets), len(testers)))
	print classifier.show_most_informative_features()
	print "This classifier is currently running at (percentage):", nltk.classify.accuracy(classifier, testing_set)


	print "\nTweet Sentiment Analyzer\n\tBy Patrick Gaston\n\n\n\n"
	promptuser = raw_input("Enter your tweet here: ")
	print ("The input is considered: ", classifier.classify(extract_feats(promptuser.split())))

	prompt_again = raw_input("\n\nTry another tweet: ")
	print ("The input is considered: ", classifier.classify(extract_feats(prompt_again.split())))

	prompt_last = raw_input("\n\nYou know what... Try another tweet: ")
	print ("The input is considered: ", classifier.classify(extract_feats(prompt_last.split())))

	run_again = raw_input("\n\nWould you like to enter 3 more tweets? Enter (Y/N) ")
	if run_again == "Y":
		return main()
	else:
		print "Bye!"


main()
