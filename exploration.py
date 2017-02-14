'''
Let's get all we need

Let's start by importing all libraries we need, read the input CSV and check everything is loaded by looking at all the different speakers and dates of the debates.
'''

# Import libraries
import pandas as pd

import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet
from sklearn.feature_extraction import stop_words

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer 

import matplotlib.pyplot as plt
import seaborn as sns

# Read CSV
path = 'debate.csv'
df = pd.read_csv(path, encoding = 'ISO-8859-1')

# Print speakers and dates
print(df['Speaker'].unique())
print(df['Date'].unique())

'''
Let's define the different parameters we are going to use for the different plots.
'''

candidates = ['Clinton', 'Trump']
all_dates = df[df['Speaker'].isin(candidates)]['Date'].unique()

colors = dict(zip(candidates, ['#3498db', '#e74c3c']))
print(colors)

line_styles = dict(zip(all_dates, ['solid', 'dashed', 'dotted']))
print(line_styles)

'''
Words by response

Let's start by a comparison of the number of words by response.

For counting the number of words, I decided to use the 'RegexpTokenizer' from 'nltk.tokenize'. This does job well though it is not perfect. For example, if there are spaces before and after a punctuation mark, that punctuation mark will be count as a word. But this should not happen too often.
'''

# Create a tokenizer for counting the number of words
tokenizer = RegexpTokenizer('\s+', gaps=True)

# Count words
df['Word_count'] = df['Text'].map(lambda x: len(tokenizer.tokenize(x)))

# Show box plot of distribution
plt.figure()
sns.boxplot(x='Date', y='Word_count', hue='Speaker', data=df[df['Speaker'].isin(candidates)], palette=colors)
plt.title('Number of words by response')
plt.show()

# Show statistics
response_stats = df[df['Speaker'].isin(candidates)].groupby(['Speaker', 'Date']).agg({'Word_count': ['count', 'mean']})
print(response_stats)

'''
You can notice there are quite some differences between debates. This could be a reflection of how each presenter controls the debate.

But, generally speaking, Trump has in average shorter responses while at the same time he has a higher number of responses.
'''

# Plot distributions
fig = plt.figure()
for i in range(0, len(all_dates)):
    ax = fig.add_subplot(1,len(all_dates),i + 1)    
    for j in range(0,len(candidates)):
        sns.distplot(df[(df['Speaker'] == candidates[j]) & (df['Date']== all_dates[i])]['Word_count'], color=colors[candidates[j]])
    ax.set_title(all_dates[i])
plt.show()

'''
Words by sentence

Like I said my idea was that the average number of words by sentence could be a way for differentiating speakers. I wanted to check if that was the case or not.
'''

# Count number of words by sentence and debate
def split_in_sentence(group):
    w_by_s = []
    speaker, date = group.name
    for t in group['Text']:
        sentences = nltk.sent_tokenize(t)
        Word_count = [len(tokenizer.tokenize(s)) for s in sentences]
        w_by_s = w_by_s + Word_count
    return pd.DataFrame({'Speaker': speaker, 'Date': date , 'Word_count': w_by_s })    

w_by_s_df = df[df['Speaker'].isin(candidates)].groupby(['Speaker', 'Date']).apply(split_in_sentence)

# Print statistics over the number of sentences
w_by_s_stats = w_by_s_df.groupby(['Speaker', 'Date']).agg({'Word_count': ['count', 'mean']})
print(w_by_s_stats)

'''
You can notice here that the averages for a particular candidate are close to each other, while the averages of the two candidates seem to be 'significantly' different. Let's now check that visually by plotting the KDE's that can derived from the data.
'''

# Plot number of words by sentence
plt.figure()
sns.violinplot(x='Date', y = 'Word_count', hue='Speaker', data=w_by_s_df, palette=colors, split= True)
plt.title('Number of words by sentence')
plt.show()

# Plot number of words by sentence
plt.figure()
sns.boxplot(x='Speaker', y='Word_count', data=w_by_s_df, palette=colors)
plt.show()

# Plot number of words by sentence
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
for i in range(0,len(candidates)):
        #sns.distplot(w_by_s_df[w_by_s_df['Speaker']==candidates[i]]['Word_count'], color=colors[candidates[i]], rug=True, hist=False)
        sns.kdeplot(w_by_s_df[w_by_s_df['Speaker']==candidates[i]]['Word_count'], color=colors[candidates[i]], shade = True)
h, l = ax.get_legend_handles_labels()
ax.legend(h, candidates)
plt.show()

# Plot KDE's
def plot_kde(group):
    c,d = group.name
    sns.kdeplot(group['Word_count'], color=colors[c], linestyle=line_styles[d])
    return c + ' on ' + d

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
legends = w_by_s_df[w_by_s_df['Speaker'].isin(candidates)].groupby(['Speaker', 'Date']).apply(plot_kde)
h, l = ax.get_legend_handles_labels()
ax.legend(h, legends)
plt.title('Number of words by sentence (KDE only)')
plt.show()

'''
Cool ! It seems the number of words by sentence is a way to differentiate speakers ! But OK, this might be dependent on the transcript.
'''

'''
Count and TF-IDF of words

In this second part, I extract the most important words and bi-grams of the debate by doing a count and/or a TF-IDF.

I have added some stop words iteratively to scikit-learn's list of stop words based on the output I got. By the way there is a good list of stop words available at http://members.unine.ch/jacques.savoy/clef/englishST.txt.

While working on it, I noticed the word 'isi' was appearing in the top features. The reason was that word 'ISIS' was lemmatized into 'isi'. For avoiding that, I do not lower words in uppercase (e.g. NATO, ISIS) before lemmatization.

Being mostly interested in the 'topics' of the debate, I also decided to remove the verbs from during the text processing part. 
'''

# Top n features we want to display
top_n = 20

# Stop words
s_w = set(stop_words.ENGLISH_STOP_WORDS)
extra_s_w = ['ahead', 'back', 'bad', 'big', 'billion', 'bit', 'bring', 'call', 'clear', 'clearly', 'come', 'down', 'fact', 'good', 'great', 'happen', 'hear', 'important', 'inner', 'just', 'kind', 'know', 'like', 'little', 'look', 'lot', 'make', 'maybe', 'million', 'new', 'percent', 'people', 'put', 'question', 'really', 'right', 'say', 'somebody', 'stand', 'start', 'stop', 'talk', 'tell', 'thing', 'think', 'time', 'trillion', 'wait', 'want', 'way', 'word', 'wrong', 'year']
s_w = s_w.union(extra_s_w)
    
# Map POS tags to wordnet POS tags
# See http://www.ling.helsinki.fi/~gwilcock/Tartu-2011/P2-nltk-2.xhtml
def map_pos_to_wordnet_pos(tag):
    if tag.startswith('NN'):
        return wordnet.NOUN
    elif tag.startswith('VB'):
        return wordnet.VERB
    elif tag.startswith('JJ'):
        return wordnet.ADJ
    elif tag.startswith('RB'):
        return wordnet.ADV
    else:
        return wordnet.NOUN
    
# Function for processing the text
def process_text(data_set, excluded_pos):
    lmtzr = WordNetLemmatizer()
    transformed_data_set = []    
    for text in data_set: 
        # Tokenize
        tokens = nltk.word_tokenize(text)
        # Get part-of-speech tag        
        tagged_text = nltk.pos_tag(tokens) 
        # Keep text only
        tagged_text = [(token, pos) for (token, pos) in tagged_text if token.isalpha()]
        # Map to wordnet POS
        tagged_text = [(token, map_pos_to_wordnet_pos(pos)) for (token, pos) in tagged_text]
        # Remove POS we do not care about
        tagged_text = [(token, pos) for (token, pos) in tagged_text if pos not in excluded_pos]
        # Lemmatize
        lmtzd_tokens = []
        for token, pos in tagged_text:
            if token.isupper():
                # REMARK: for 'ISIS' or 'NATO'
                lmtzd_tokens.append(token)
            else:
                lmtzd_tokens.append(lmtzr.lemmatize(token.lower(), pos))
        # Remove stop words
        lmtzd_tokens = [token for token in lmtzd_tokens if token not in s_w]
        # Keep only words with length > 2
        lmtzd_tokens = [token for token in lmtzd_tokens if len(token) > 2]
        # Append to the summary tokens
        transformed_data_set.append(' '.join(lmtzd_tokens))
    return transformed_data_set

# Function for getting the count and the TF-IDF
def get_tfidf(text, ngram):
    count_vectorizer = CountVectorizer(ngram_range=ngram)
    x_counts = count_vectorizer.fit_transform(text)  
    count = x_counts.toarray()
    tfidf_transformer = TfidfTransformer()
    tfidf = tfidf_transformer.fit_transform(x_counts).toarray()
    return count_vectorizer, count, tfidf_transformer, tfidf

# Group text by candidate and debate
text_df = df[df['Speaker'].isin(candidates)].groupby(['Date', 'Speaker'])['Text'].apply(lambda x: ' '.join(x))
keys = [k[0] + '_' + k[1] for k in text_df.index.values]
candidate_text = text_df.tolist()

# Process text
candidate_text = process_text(candidate_text, [wordnet.VERB])

# Compute TF-IDF
count_vect, count, tfidf_transf, tfidf = get_tfidf(candidate_text, (1,1))
features = count_vect.get_feature_names()
print(len(features))

# Link features to their count and TF-IDF
tfidf_dict = { 'Features': features }
for i in range(0, len(keys)):
    tfidf_dict['Tfidf_' + keys[i]] = tfidf[i]
    tfidf_dict['Count_' + keys[i]] = count[i]
tfidf_df = pd.DataFrame(tfidf_dict)

# Print top features based on count
for i in range(0, len(keys)):
    print(tfidf_df.sort_values(by='Count_' + keys[i], ascending=False)[['Features', 'Count_' + keys[i]]].head(top_n))

# Print top features based on TF-IDF
for i in range(0, len(keys)):
    print(tfidf_df.sort_values(by='Tfidf_' + keys[i], ascending=False)[['Features', 'Tfidf_' + keys[i]]].head(top_n))
    
'''
Count and TF-IDF of words

I do the same for bi-grams as for words. This could also be done easily for N-grams.
'''

# Compute TF-IDF
ngram_count_vect, ngram_count, ngram_tfidf_transf, ngram_tfidf = get_tfidf(candidate_text, (2,2))
ngram_features = ngram_count_vect.get_feature_names()

# Link features to their count and TF-IDF
ngram_dict = { 'Features': ngram_features }
for i in range(0, len(keys)):
    ngram_dict['Tfidf_' + keys[i]] = ngram_tfidf[i]
    ngram_dict['Count_' + keys[i]] = ngram_count[i]
ngram_df = pd.DataFrame(ngram_dict)

# Print top features based on count
for i in range(0, len(keys)):
    print(ngram_df.sort_values(by='Count_' + keys[i], ascending=False)[['Features', 'Count_' + keys[i]]].head(top_n))

# Print top features based on TF-IDF
for i in range(0, len(keys)):
    print(ngram_df.sort_values(by='Tfidf_' + keys[i], ascending=False)[['Features', 'Tfidf_' + keys[i]]].head(top_n))
