import pandas as pd
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet as wn
from fuzzywuzzy import fuzz
from sklearn import svm
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


#finds longest subsequence of words in sentences
def longestSS(sentence1, sentence2):
    shorter = " "
    if (len(sentence1) < len(sentence2)):
        shorter = len(sentence1)
    else:
        shorter = len(sentence2)  
    
    traversed = 0
    stored = 0
    i = 0
    while(i < shorter):
        if sentence1[i] == sentence2[i]:
            i += 1
            traversed += 1
        else:
            if stored < traversed:
                stored = traversed
            traversed = 0
            i += 1
    return stored

#number of overlapping words
def getOverLap(s1, s2):
    overlap = 0
    words1 = set(s1.translate(str.maketrans('', '', string.punctuation)).split(" "))
    words2 = set(s2.translate(str.maketrans('', '', string.punctuation)).split(" "))

    for word in words1:
        if word in words2:
            overlap += 1
            
    return overlap

#get synonyms
def synAmount(s1, s2):
    synonyms = []
    amount = 0
    words1 = set(s1.translate(str.maketrans('', '', string.punctuation)).split(" "))
    words2 = set(s2.translate(str.maketrans('', '', string.punctuation)).split(" "))
    
    for word in words1:
        for syn in wn.synsets(word):
            for lemma in syn.lemmas():
                synonyms.append(lemma.name())
                if lemma.antonyms():
                    continue
                    
    for synonym in set(synonyms):
        if synonym in words2:
            amount += 1
    return amount

#cosine similarity
def cosSimilarity(s1, s2):
    s1_list = word_tokenize(s1)
    s2_list = word_tokenize(s2)

    sw = stopwords.words('english')
    l1 = []
    l2 = []

    s1_set = {w for w in s1_list if not w in sw}
    s2_set = {w for w in s2_list if not w in sw}

    rvector = s1_set.union(s2_set)
    for w in rvector:
        if w in s1_set: l1.append(1)
        else: l1.append(0)

        if w in s2_set: l2.append(1)
        else: l2.append(0)
    
    c = 0
    for i in range(len(rvector)):
        c += l1[i] * l2[i]
    cosine = c / (float(sum(l1) * sum(l2))**0.5)

    return int(cosine * 100)

#preprocess data
def preprocessing(txtFile):
    columns = ['id', 'sentence 1', 'sentence 2', 'gold label']
    df = pd.read_csv(txtFile, sep = '\t+', names = columns, engine='python')
    df['gold label'] = pd.to_numeric(df['gold label'], errors='coerce')
    df = df.dropna()
    df['gold label'] = df['gold label'].astype(int)

    df['amt words 1'] = df['sentence 1'].apply(lambda x: len(x.translate(str.maketrans('', '', string.punctuation)).split(" ")))
    df['amt words 2'] = df['sentence 2'].apply(lambda x: len(x.translate(str.maketrans('', '', string.punctuation)).split(" ")))

    df['longest subseq.'] = df.apply(lambda row: longestSS(row['sentence 1'].split(), row['sentence 2'].split()), axis = 1)
    df['length difference'] = abs(df['amt words 1'] - df['amt words 2'])
    df['# of overlap'] = df.apply(lambda row: getOverLap(row['sentence 1'], row['sentence 2']), axis = 1)
    df['similarity ratio'] = df.apply(lambda row: fuzz.ratio(row['sentence 1'].translate(str.maketrans('', '', string.punctuation)), row['sentence 2'].translate(str.maketrans('', '', string.punctuation))), axis = 1)
    df['synonyms'] = df.apply(lambda row: synAmount(row['sentence 1'], row['sentence 2']), axis = 1)
    df['cosine similarity'] = df.apply(lambda row: cosSimilarity(row['sentence 1'].translate(str.maketrans('', '', string.punctuation)), row['sentence 2'].translate(str.maketrans('', '', string.punctuation))), axis = 1)

    return df

def preprocesstest(txtFile):
    test_columns = ['instance id', 'sentence 1', 'sentence 2']
    df = pd.read_csv(txtFile, sep = '\t+', names = test_columns, engine='python')
    df['sentence 1'] = df['sentence 1'].astype(str)
    df['sentence 2'] = df['sentence 2'].astype(str)
    df['amt words 1'] = df['sentence 1'].apply(lambda x: len(x.translate(str.maketrans('', '', string.punctuation)).split(" ")))
    df['amt words 2'] = df['sentence 2'].apply(lambda x: len(x.translate(str.maketrans('', '', string.punctuation)).split(" ")))

    df['longest subseq.'] = df.apply(lambda row: longestSS(row['sentence 1'].split(), row['sentence 2'].split()), axis = 1)
    df['length difference'] = abs(df['amt words 1'] - df['amt words 2'])
    df['# of overlap'] = df.apply(lambda row: getOverLap(row['sentence 1'], row['sentence 2']), axis = 1)
    df['similarity ratio'] = df.apply(lambda row: fuzz.ratio(row['sentence 1'].translate(str.maketrans('', '', string.punctuation)), row['sentence 2'].translate(str.maketrans('', '', string.punctuation))), axis = 1)
    df['synonyms'] = df.apply(lambda row: synAmount(row['sentence 1'], row['sentence 2']), axis = 1)
    df['cosine similarity'] = df.apply(lambda row: cosSimilarity(row['sentence 1'].translate(str.maketrans('', '', string.punctuation)), row['sentence 2'].translate(str.maketrans('', '', string.punctuation))), axis = 1)

    return df

#Declare Dataframes
df_train = preprocessing('training.txt')
df_dev = preprocessing('dev.txt')
df_test = preprocesstest('test.txt')

X_train = df_train.iloc[:,6:]
y_train = df_train['gold label']
X_dev = df_dev.iloc[:,6:]
y_dev = df_dev['gold label']

SVM_classifier = make_pipeline(StandardScaler(), svm.SVC(kernel = 'rbf', gamma = 1, C = 1, class_weight = 'balanced'))
SVM_classifier.fit(X_train, y_train)

test_features = ['longest subseq.', 'length difference', '# of overlap', 'similarity ratio', 'synonyms', 'cosine similarity']
X_test = df_test[test_features]
y_test_pred = SVM_classifier.predict(X_test)


file = open('AnubhavKundu_test_result.txt', 'w')
for i in range(len(df_test['instance id'])):
    file.write(str(df_test['instance id'].values[i]) + '\t' + str(y_test_pred[i]) + '\n')
file.close()







