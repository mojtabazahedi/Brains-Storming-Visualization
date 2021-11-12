#!/usr/bin/env python
# coding: utf-8

# In[1]:


from gensim.models import Word2Vec, KeyedVectors


# In[2]:


model = KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin.gz", binary=True)


# In[3]:


keys = ['game', 'app','add', 'play', 'players', 'try', 'different', 'guess',
        'gaming', 'application', 'features', 'select', 'children',
        'like', 'chance', 'ball', 'attempt', 'person', 'line', 'user',
        'great', 'developing', 'option', 'provide', 'users', 'race', 'popular','card',
        'people', 'teams', 'app', 'test', 'virtual', 'surely', 'mobile', 'consider', 'kids',
        'creating', 'making', 'stand', 'finish', 'turns', 'characters', 'exciting', 'way', 'racing', 'choose',
       'adding','develop','help','attractive','images', 'right', 'clear', 'instructions', 'trains', 
        'favorite', 'played', 'cricket','playing', 'grab', 'attention', 'piece', 'paper', 'hands', 'music', 
        'saying', 'crime', 'entice', 'character', 'story', 'sports', 'sport', 'best', 'improve', 'adults', 'ones', 
        'google', 'challenging', 'compete', 'animal', 'choice', 'reality', 'everyone', 'set', 'counting',
        'alphabets', 'children', 'think','aware', 'activities', 'driving', 'big', 'different', 'especially','hit',
        'need', 'feature', 'potential', 'poker', 'joining', 'rewards', 'dealer', 'exceptional', 'game', 'intellectual',
       'simple', 'start', 'match', 'puzzle', 'features', 'sure', 'questions', 'levels', 'create', 'famous', 'worldwide',
        'player', 'spin', 'stops', 'makes', 'virtually', 'team', 'baseball', 'the', 'understand', 'ludo',
        'touch', 'trying', 'wins', 'figure', 'paper', 'bat', 'sit', 'circle', 'child', 'blindfolded', 'time', 'bag',
        'objects', 'thing','don', 'eat', 'pass', 'caught', 'egg','f1','motogp','hamilton','nascar','wrc','basketball',
        'lebron','football','socer']
for i in keys :
    vector = model[i]
    print(i, vector[:5])
    


# In[4]:


keys = ['game', 'app','add', 'play', 'players', 'try', 'different', 'guess',
        'player', 'gaming', 'application', 'features', 'select', 'children',
        'like', 'chance', 'ball', 'attempt', 'person', 'line', 'games', 'user',
        'great', 'developing', 'option', 'provide', 'users', 'race', 'popular','card',
        'people', 'teams', 'app', 'test', 'virtual', 'surely', 'mobile', 'consider', 'kids',
        'creating', 'making', 'stand', 'finish', 'turns', 'characters', 'exciting', 'way', 'racing', 'choose']

embedding_clusters = []
word_clusters = []
for word in keys:
    embeddings = []
    words = []
    for similar_word, _ in model.most_similar(word, topn=30):
        words.append(similar_word)
        embeddings.append(model[similar_word])
    embedding_clusters.append(embeddings)
    word_clusters.append(words)


# In[6]:


from sklearn.manifold import TSNE


# In[7]:


import numpy as np


# In[8]:


embedding_clusters = np.array(embedding_clusters)
n, m, k = embedding_clusters.shape
tsne_model_en_2d = TSNE(perplexity=15, n_components=2, init='pca', n_iter=3500, random_state=32)
embeddings_en_2d = np.array(tsne_model_en_2d.fit_transform(embedding_clusters.reshape(n * m, k))).reshape(n, m, 2)


# In[9]:


import matplotlib.pyplot as plt
import matplotlib.cm as cm
get_ipython().run_line_magic('matplotlib', 'inline')


# In[11]:


def tsne_plot_similar_words(title, labels, embedding_clusters, word_clusters, a, filename=None):
    plt.figure(figsize=(16, 9))
    colors = cm.rainbow(np.linspace(0, 1, len(labels)))
    for label, embeddings, words, color in zip(labels, embedding_clusters, word_clusters, colors):
        x = embeddings[:, 0]
        y = embeddings[:, 1]
        plt.scatter(x, y, c=color, alpha=a, label=label)
        for i, word in enumerate(words):
            plt.annotate(word, alpha=0.5, xy=(x[i], y[i]), xytext=(5, 2),
                         textcoords='offset points', ha='right', va='bottom', size=8)
    plt.legend(loc=4)
    plt.title(title)
    plt.grid(True)
    if filename:
        plt.savefig(filename, format='png', dpi=500, bbox_inches='tight')
    plt.show()


tsne_plot_similar_words('Similar words from Google News', keys, embeddings_en_2d, word_clusters, 0.7,
                        'similar_words.png')


# In[61]:


key = ['game', 'app','add', 'play', 'players', 'try', 'different', 'guess',
        'gaming', 'application', 'features', 'select', 'children',
        'like', 'chance', 'ball', 'attempt', 'person', 'line', 'user',
        'great', 'developing', 'option', 'provide', 'users', 'race', 'popular','card',
        'people', 'teams', 'app', 'test', 'virtual', 'surely', 'mobile', 'consider', 'kids',
        'creating', 'making', 'stand', 'finish', 'turns', 'characters', 'exciting', 'way', 'racing', 'choose',
       'adding','develop','help','attractive','images', 'right', 'clear', 'instructions', 'trains', 
        'favorite', 'played', 'cricket','playing', 'grab', 'attention', 'piece', 'paper', 'hands', 'music', 
        'saying', 'crime', 'entice', 'character', 'story', 'sports', 'sport', 'best', 'improve', 'adults', 'ones', 
        'google', 'challenging', 'compete', 'animal', 'choice', 'reality', 'everyone', 'set', 'counting',
        'alphabets', 'children', 'think','aware', 'activities', 'driving', 'big', 'different', 'especially','hit',
        'need', 'feature', 'potential', 'poker', 'joining', 'rewards', 'dealer', 'exceptional', 'game', 'intellectual',
       'simple', 'start', 'match', 'puzzle', 'features', 'sure', 'questions', 'levels', 'create', 'famous', 'worldwide',
        'player', 'spin', 'stops', 'makes', 'virtually', 'team', 'baseball', 'the', 'understand', 'ludo',
        'touch', 'trying', 'wins', 'figure', 'paper', 'bat', 'sit', 'circle', 'child', 'blindfolded', 'time', 'bag',
        'objects', 'thing','don', 'eat', 'pass', 'caught', 'egg','f1','motogp','hamilton','nascar','basketball',
        'lebron','football']
keys = list(set(key))
def tsne_plot(model):
    "Creates and TSNE model and plots it"
    labels = []
    tokens = []
    
    for word in keys :
        tokens.append(model[word])
        labels.append(word)
        
    
    
    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=3500, random_state=32)
    new_values = tsne_model.fit_transform(tokens)

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])
        
    plt.figure(figsize=(18, 18)) 
    for i in range(len(x)):
        plt.scatter(x[i],y[i])
        plt.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.grid(True)
    plt.title("WordVector Presentation")
    #plt.legend(labels, loc='upper center', bbox_to_anchor=(0.5, 1.05),ncol=15, fancybox=True, shadow=True)
    plt.show()


# In[62]:


tsne_plot(model)


# In[ ]:




