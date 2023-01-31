#!/usr/bin/env python
# coding: utf-8

# In[23]:


import io
import csv
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt


# In[24]:


with open("./bbc-text.csv", 'r') as csvfile:
    print(f"First line (header) looks like this:\n\n{csvfile.readline()}")
    print(f"Each data point looks like this:\n\n{csvfile.readline()}")


# - `NUM_WORDS`: Sözcük sıklığına bağlı olarak tutulacak maksimum sözcük sayısı. Varsayılanlar 1000'dir.
# 
# 
# - `EMBEDDING_DIM`: Yoğun gömme boyutu, modelin gömme katmanında kullanılacaktır. Varsayılanlar 16'dır.
# 
# 
# - `MAXLEN`: Tüm dizilerin maksimum uzunluğu. Varsayılanlar 120'dir.
# 
# 
# - `PADDING`: Doldurma stratejisi (her diziden önce veya sonra dolgu yapın.). Varsayılanlar 'gönder'dir.
# 
# 
# - `OOV_TOKEN`: text_to_sequence çağrıları sırasında kelime dağarcığı dışındaki kelimeleri değiştirmek için belirteç. Varsayılanlar "<OOV>" şeklindedir.
# 
#     
# - `TRAINING_SPLIT`: Eğitim için kullanılan verilerin oranı. Varsayılanlar 0,8'dir
# 

# In[25]:


NUM_WORDS = 1000
EMBEDDING_DIM = 16
MAXLEN = 120
PADDING = 'post'
OOV_TOKEN = "<OOV>"
TRAINING_SPLIT = .8


# In[26]:


'''
Metinden engellenecek sözcükleri kaldırmak ve verileri
bir csv dosyasından yüklemek için kullanılan kod.
'''

def remove_stopwords(sentence):
    """
    Removes a list of stopwords
    
    Args:
        sentence (string): sentence to remove the stopwords from
                         : stopwords kaldırmak için cümle
                          
    Returns:
        sentence (string): lowercase sentence without the stopwords
                         : stopwords olmadan küçük harfli cümle
    """
    # List of stopwords
    stopwords = ["a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from", "further", "had", "has", "have", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "it", "it's", "its", "itself", "let's", "me", "more", "most", "my", "myself", "nor", "of", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "she", "she'd", "she'll", "she's", "should", "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "we", "we'd", "we'll", "we're", "we've", "were", "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "would", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves" ]
    
    # Sentence converted to lowercase-only
    #cümleler küçük harflere dönüştürüldü
    sentence = sentence.lower()
    
#     print("sentence: \n\n", sentence)

    # kelime kelime parçaladı
    words = sentence.split()
    
#     print("split word: ", words)
    
    no_words = [w for w in words if w not in stopwords]
    sentence = " ".join(no_words)
#     print("sentence: \n\n", sentence)

    return sentence

def parse_data_from_file(filename):
    """
    Extracts sentences and labels from a CSV file
    
    Args:
        filename (string): path to the CSV file
    
    Returns:
        sentences, labels (list of string, list of string):
        tuple containing lists of sentences and labels
    """
    sentences = []
    labels = []
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader)
        for row in reader:
            labels.append(row[0])
            sentence = row[1]
            sentence = remove_stopwords(sentence)
            sentences.append(sentence)

    return sentences, labels

sentences, labels = parse_data_from_file("./bbc-text.csv")

print(f"There are {len(sentences)} sentences in the dataset.\n")
print(f"First sentence has {len(sentences[0].split())} words (after removing stopwords).\n")
print(f"There are {len(labels)} labels in the dataset.\n")
print(f"The first 5 labels are {labels[:5]}")
print("\n\n")


# In[27]:


# GRADED FUNCTIONS: train_val_split
def train_val_split(sentences, labels, training_split):
    """
    Splits the dataset into training and validation sets
    
    Args:
        sentences (list of string): lower-cased sentences without stopwords
        labels (list of string): list of labels
        training split (float): proportion of the dataset to convert to include in the train set
        
        sentences (dize listesi): stopwords içermeyen küçük harfli cümleler
        labels (dize listesi): etiket listesi
        training split (kayan nokta): eğitim setine dahil edilecek
        dönüştürülecek veri setinin oranı
        
    Returns:
        train_sentences, validation_sentences, train_labels, validation_labels - lists containing the data splits
    """
    
    # Compute the number of sentences that will be used for training (should be an integer)
    train_size = int(len(sentences) * training_split) # training_split = 0.8

    # Split the sentences and labels into train/validation splits
    train_sentences = sentences[:train_size]
    train_labels = labels[:train_size]

    validation_sentences = sentences[train_size:]
    validation_labels = labels[train_size:]
    
    
    return train_sentences, validation_sentences, train_labels, validation_labels

# Test function
train_sentences, val_sentences, train_labels, val_labels = train_val_split(sentences, labels, TRAINING_SPLIT)

print(f"There are {len(train_sentences)} sentences for training.\n")
print(f"There are {len(train_labels)} labels for training.\n")
print(f"There are {len(val_sentences)} sentences for validation.\n")
print(f"There are {len(val_labels)} labels for validation.")


# In[28]:


# GRADED FUNCTION: fit_tokenizer
def fit_tokenizer(train_sentences, num_words, oov_token):
    """
    Instantiates the Tokenizer class on the training sentences
    
    Args:
        train_sentences (list of string): lower-cased sentences without stopwords to be used for training
        num_words (int) - number of words to keep when tokenizing
        oov_token (string) - symbol for the out-of-vocabulary token
        
        train_sentences (list of string): eğitim için kullanılacak stopwords
        içermeyen küçük harfli cümleler
        num_words (int) - belirteç oluştururken saklanacak sözcük sayısı
        oov_token (string) - kelime dağarcığı dışı belirteci için sembol      
    
    Returns:
        tokenizer (object): an instance of the Tokenizer class containing the word-index dictionary
    """
    
    # Instantiate the Tokenizer class, passing in the correct values for num_words and oov_token
    tokenizer = Tokenizer(num_words = num_words, oov_token = oov_token )
    
    print("tokenizer: \n", tokenizer, "\n")
    
    # Fit the tokenizer to the training sentences
    tokenizer.fit_on_texts(train_sentences)
    print("tokenizer.fit_on_texts: \n", tokenizer.fit_on_texts(train_sentences), "\n")
    
    
    return tokenizer

"""
NUM_WORDS = 1000
EMBEDDING_DIM = 16
MAXLEN = 120
PADDING = 'post'
OOV_TOKEN = "<OOV>"
TRAINING_SPLIT = .8
"""

# Test function
tokenizer = fit_tokenizer(train_sentences, NUM_WORDS, OOV_TOKEN) 
print("fit_tokenizer: \n", tokenizer, "\n")
word_index = tokenizer.word_index
print("first 10 dict word_index: \n", list(word_index.items())[:10], "\n")

print(f"Vocabulary contains {len(word_index)} words\n")
print("<OOV> token included in vocabulary" if "<OOV>" in word_index else "<OOV> token NOT included in vocabulary")


# In[29]:


6# GRADED FUNCTION: seq_and_pad
def seq_and_pad(sentences, tokenizer, padding, maxlen):
    """
    Generates an array of token sequences and pads them to the same length
    Bir dizi belirteç dizisi oluşturur ve bunları aynı uzunlukta doldurur
    
    Args:
        sentences (list of string): list of sentences to tokenize and pad
        tokenizer (object): Tokenizer instance containing the word-index dictionary
        padding (string): type of padding to use
        maxlen (int): maximum length of the token sequence
    
    Returns:
        padded_sequences (array of int): tokenized sentences padded to the same length
    """    
       
    # Convert sentences to sequences
    # Cümleleri dizilere dönüştür
    sequences = tokenizer.texts_to_sequences(sentences)
    
    # Pad the sequences using the correct padding and maxlen
    # Doğru dolgu ve maxlen kullanarak dizileri doldurun
    padded_sequences = pad_sequences(sequences, padding = padding,maxlen = maxlen)

    
    return padded_sequences

"""
NUM_WORDS = 1000
EMBEDDING_DIM = 16
MAXLEN = 120
PADDING = 'post'
OOV_TOKEN = "<OOV>"
TRAINING_SPLIT = .8
"""
# Test  function
train_padded_seq = seq_and_pad(train_sentences, tokenizer, PADDING, MAXLEN)
val_padded_seq = seq_and_pad(val_sentences, tokenizer, PADDING, MAXLEN)

print(f"Padded training sequences have shape: {train_padded_seq.shape}\n")
print(f"Padded validation sequences have shape: {val_padded_seq.shape}")


# * Etiketleri belirtilmeli. Unutulmaması gereken birkaç nokta: Doğrulama setinde belirli bir etiketin bulunmaması durumundan kaçınmak için belirteç oluşturucuyu tüm etiketlere sığdırılmalı.
# * Etiketlerle uğraşıldığı için asla bir OOV etiketi olmamalıdır. Önceki işlevde, sayısal diziler döndüren pad_sequences işlevini kullanıldı. Etiketlerin doldurulması gerekmediğinden burada onu kullanmayacağız, bu nedenle numpy dizilerine dönüştürmeyi kendiniz yapmanız gerekir. split_labels bağımsız değişkeni, belirli bir ayırmanın (tren veya doğrulama) etiketlerine atıfta bulunur. 
# * Bunun nedeni, işlevin kullanılan bölmeden bağımsız çalışması gerektiğidir. Keras'ın Tokenizer'ı kullanmak, 0'dan ziyade 1'den başlayan değerler verir. Keras genellikle etiketlerin 0'dan başlamasını beklediğinden, eğitim sırasında bu bir sorun teşkil eder. Bu soruna geçici bir çözüm bulmak için, son katmanınızda fazladan bir nöron kullanılabilir. modeli. Ancak bu yaklaşım oldukça hileli ve çok net değil. Bunun yerine, işlevin döndürdüğü etiketlerin her değerinden 1 çıkaracaksınız. Numpy dizilerini kullanırken, numpy vektörleştirilmiş işlemlere izin verdiğinden, bunu gerçekleştirmek için basitçe np.array - 1 gibi bir şey yapalabilir.

# In[30]:


# GRADED FUNCTION: tokenize_labels
def tokenize_labels(all_labels, split_labels):
    """
    Tokenizes the labels
    
    Args:
        all_labels (list of string): labels to generate the word-index from
        split_labels (list of string): labels to tokenize
        
        all_labels (list of string): kelime indeksini oluşturmak için etiketler
    
    Returns:
        label_seq_np (array of int): tokenized labels
    """
    sentences = []
    for t in split_labels:
        if t in all_labels:
            sentences.append(t)
    
    # Instantiate the Tokenizer (no additional arguments needed)
    label_tokenizer = Tokenizer()
    
    # Fit the tokenizer on all the labels
    label_tokenizer.fit_on_texts(all_labels)
    
    # Convert labels to sequences
    label_seq = np.array(label_tokenizer.texts_to_sequences(split_labels))
    validation_label_seq = np.array(label_tokenizer.texts_to_sequences(split_labels))

    
    # Convert sequences to a numpy array. Don't forget to substact 1 from every entry in the array!
    label_seq_np =np.array([number - 1 for number in label_seq]) 

    
    return label_seq_np

# Test  function
train_label_seq = tokenize_labels(labels, train_labels)
val_label_seq = tokenize_labels(labels, val_labels)

print(f"First 5 labels of the training set should look like this:\n{train_label_seq[:5]}\n")
print(f"First 5 labels of the validation set should look like this:\n{val_label_seq[:5]}\n")
print(f"Tokenized labels of the training set have shape: {train_label_seq.shape}\n")
print(f"Tokenized labels of the validation set have shape: {val_label_seq.shape}\n")


# ## Metin sınıflandırması için model seçme
# 
# 
# - Bu işlevin, tümü bir [Embedding](https://www.tensorflow.org/api_docs/python/tf/keras/) iletilmesi amaçlanan üç parametreye sahip olduğuna dikkat edilmeli. katmanlar/Gömme) katmanı,  muhtemelen model için ilk katman olarak kullanılankatmandır.
# 
# - Son katman, softmax aktivasyonu ile 5 birimli (5 kategori olduğundan) Yoğun bir katman olmalıdır. - Model uygun bir kayıp fonksiyonu ve iyileştirici kullanarak da derlenmeli.
# 
# (https://www.tensorflow.org/api_docs/python/tf/keras/layers/GlobalAveragePooling1D) ve Yoğun katmanların yanı sıra herhangi bir katmana ihtiyaç yoktur, ancak farklı mimariler denenebilir.
# 
# - **Bu kademeli işlevi geçmek için modelin 30 dönemin altında en az %95 eğitim doğruluğuna ve %90 doğrulama doğruluğuna ulaşması gerekir.**

# In[31]:


# GRADED FUNCTION: create_model
def create_model(num_words, embedding_dim, maxlen):
    """
    Creates a text classifier model
    
    Args:
        num_words (int): size of the vocabulary for the Embedding layer input
        embedding_dim (int): dimensionality of the Embedding layer output
        maxlen (int): length of the input sequences
    
    Returns:
        model (tf.keras Model): the text classifier model
    """
    
    tf.random.set_seed(123)
    

    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(num_words, embedding_dim, input_length=maxlen),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(24, activation='relu'),
        tf.keras.layers.Dense(6, activation='softmax')
    ])
    model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])


    return model


model = create_model(NUM_WORDS, EMBEDDING_DIM, MAXLEN)
history = model.fit(train_padded_seq, train_label_seq, epochs=30, validation_data=(val_padded_seq, val_label_seq))

def plot_graphs(history, metric):
    plt.plot(history.history[metric])
    plt.plot(history.history[f'val_{metric}'])
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend([metric, f'val_{metric}'])
    plt.show()
    
plot_graphs(history, "accuracy")
plot_graphs(history, "loss")


# In[ ]:




