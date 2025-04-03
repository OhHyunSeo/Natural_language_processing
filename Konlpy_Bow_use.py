#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 11 11:29:31 2025

@author: oh
"""

import nltk
nltk.download('stopwords')

from konlpy.tag import Okt

okt = Okt()

# Bag of Words 함수
# 입력된 문서에 대해서 단어 집합을 만들어 각 단어에 정수 인덱스를 할당하고, BoW

def build_bag_of_words(document):
    document = document.replace('.', '')
    tokenized_document = okt.morphs(document)
    
    word_to_index = {}
    bow = []
    
    for word in tokenized_document:
        if word not in word_to_index.keys():
            word_to_index[word] = len(word_to_index)
            bow.insert(len(word_to_index) - 1, 1)
            
        else:
            index = word_to_index.get(word)
            bow[index] = bow[index] + 1
            
    return word_to_index, bow

doc1 = '정부가 발표하는 물가상승률과 소비자가 느끼는 물가상승률은 다르다'
vocab, bow = build_bag_of_words(doc1)

print("vocab : ", vocab)
'''
vocab :  {'정부': 0, '가': 1, '발표': 2, '하는': 3, '물가상승률': 4, '과': 5, '소비자': 6, '느끼는': 7, '은': 8, '다르다': 9}
'''

print("bag of words : ", bow)
# bag of words :  [1, 2, 1, 1, 2, 1, 1, 1, 1, 1]


doc2 = '소비자는 주로 소비하는 상품을 기준으로 물가상승률을 느낀다.'
vocab, bow = build_bag_of_words(doc2)

print("vocab : ", vocab)
'''
vocab :  {'소비자': 0, '는': 1, '주로': 2, '소비': 3, '하는': 4, '상품': 5, '을': 6, '기준': 7, '으로': 8, '물가상승률': 9, '느낀다': 10}
'''

print("bag of words : ", bow)
# bag of words :  [1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1]

'''
BoW는 각 단어가 등장한 횟수를 수치화하는 텍스트 표현 방법이므로
주로 어떤 단어가 얼마나 등장했는지를 기준으로 문서가 어떤 성격의 문서인지를 판단하는 작업에 쓰인다.

즉, 분류 문제나 여러 문서 간의 유사도를 구하는 문제에 주로 쓰인다.
예)
'달리기', '체력', '근력'과 같은 단어가 자주 등장하면 해당 문서를 체육 관련 문서로 분류할 수 있을 것이며,
'미분', '방정식', '부등식'과 같은 단어가 자주 등장한다면 수학 관련 문서로 분류할 수 있다.
'''

# CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer

corpus = ['you know I want your love. Because I love you']

vector = CountVectorizer()

# 각 단어의 빈도수를 기록
print('bag of words vector : ', vector.fit_transform(corpus).toarray())
'''
vector.fit_transform(corpus).toarray())
bag of words vector :  [[1 1 2 1 2 1]]
'''

# 각 단어의 인덱스가 어떻게 부여되었는지를 출력
print("vocabulary : ", vector.vocabulary_)
'''
vocabulary :  {'you': 4, 'know': 1, 'want': 3, 'your': 5, 'love': 2, 'because': 0}
'''
'''
you와 love는 두번씩 언급되었으므로 각각 인덱스 2와 인덱스4에서 2의 

알파벳 I는 BoW를 만드는 과정에서 사라졌는데
CounterVectorizer가 기본적으로 길이가 2이상인 문자에 대해서만 토큰으로 인식하기 때문
'''

corpus = ['정부가 발표하는 물가상승률과 소비자가 느끼는 물가상승률은 다르다']
vector = CountVectorizer()

# 각 단어의 빈도수를 기록
print('bag of words vector : ', vector.fit_transform(corpus).toarray())
'''
vector.fit_transform(corpus).toarray())
bag of words vector :  [[1 1 1 1 1 1 1]]
'''

# 각 단어의 인덱스가 어떻게 부여되었는지를 출력
print("vocabulary : ", vector.vocabulary_)
'''
vocabulary :  {'정부가': 6, '발표하는': 4, '물가상승률과': 2, 
               '소비자가': 5, '느끼는': 0, '물가상승률은': 3, '다르다': 1}
'''
'''
CounterVectorizer는 띄어쓰기를 기준으로 분리한 뒤에 '물가상승률과'와 '물가상승률은'으로 조사를 포함해서 하나의 단어로
판단하기 때문에 서로 다른 두 단어로 인식
'''

# 불용어를 제거하 BoW
'''
영어의 BoW를 만들기 위해 사용하는 CounterVectorizer는 불용어를 지정하면,
불용어는 제외하고 BoW를 만들 수 있도록 불용어 제거 기능을 지원
'''
from nltk.corpus import stopwords

text = ["Family is not an important thing. It's everything."]

vect = CountVectorizer(stop_words = ["the", "a", "an", "is", "not"])

print('bag of words vector : ', vect.fit_transform(text).toarray())
'''
vect.fit_transform(text).toarray())
bag of words vector :  [[1 1 1 1 1]]
'''

# 각 단어의 인덱스가 어떻게 부여되었는지를 출력
print("vocabulary : ", vect.vocabulary_)
'''
vocabulary :  {'family': 1, 'important': 2, 'thing': 4, 'it': 3, 'everything': 0}
'''

# 자체 불용어 사용
vect = CountVectorizer(stop_words='english')
print('bag of words vector : ', vect.fit_transform(text).toarray())
'''
vect.fit_transform(text).toarray())
bag of words vector :  [[1 1 1]]
'''

print("vocabulary : ", vect.vocabulary_)
# vocabulary :  {'family': 0, 'important': 1, 'thing': 2}

# nltk에서 지원하는 불용어 사용
stop_words = stopwords.words('english')

vect = CountVectorizer(stop_words=stop_words)

print('bag of words vector : ', vect.fit_transform(text).toarray())
'''
vect.fit_transform(text).toarray())
bag of words vector :  [[1 1 1 1]]
'''
print("vocabulary : ", vect.vocabulary_)
'''
vocabulary :  {'family': 1, 'important': 2, 'thing': 3, 'everything': 0}
'''
'''
TF-IDF : TF와 IDF를 곱한값
문서를 d, 단어를 t, 문서의 총 개수를 n이라고 표현할때는 TF, DF, IDF는 
    TF : 특정 문서 d에서의 특정 단어t의 등장 횟수
    DF : 특정 단어 t가 등장한 문서의 수
    IDF : df(t)에 반비례하는 수. : log가 필요
'''

from math import log
import pandas as pd

docs = [
        '먹고 싶은 사과',
        '먹고 싶은 바나나',
        '길고 노란 바나나 바나나',
        '저는 과일이 좋아요']

vocab = list(set(w for doc in docs for w in doc.split()))
vocab.sort()

print("단어장의 크기 : ",len(vocab)) # 단어장의 크기 :  9

# TF, IDF, TF-IDF를 구하는 함수
N = len(docs)

def tf(t, d):
    return d.count(t)

def idf(t):
    df = 0
    for doc in docs:
        df += t in doc
    return log(N/(df + 1))

def tfidf(t, d):
    return tf(t, d) * idf(t)

# DTM을 데이터프레임에 저장하여 출력
result = []

for i in range(N):
    result.append([])
    
    d = docs[i]
    for j in range(len(vocab)):
        t = vocab[j]
        result[-1].append(tf(t, d))

tf_ = pd.DataFrame(result, columns= vocab)
'''
   과일이  길고  노란  먹고  바나나  사과  싶은  저는  좋아요
0    0   0   0   1    0   1   1   0    0
1    0   0   0   1    1   0   1   0    0
2    0   1   1   0    2   0   0   0    0
3    1   0   0   0    0   0   0   1    1
'''

# 각 단어에 대한 IDF 값
result = []

for j in range(len(vocab)):
    t = vocab[j]
    result.append(idf(t))
    
idf_ = pd.DataFrame(result, index = vocab, columns= ['IDF'])
'''
          IDF
과일이  0.693147
길고   0.693147
노란   0.693147
먹고   0.287682
바나나  0.287682
사과   0.693147
싶은   0.287682
저는   0.693147
좋아요  0.693147
'''
# TF-IDF 행렬
result = []

for i in range(N):
    result.append([])
    
    d = docs[i]
    for j in range(len(vocab)):
        t = vocab[j]
        result[-1].append(tfidf(t, d))
        
tfidf_ = pd.DataFrame(result, columns= vocab)
'''
        과일이        길고        노란  ...        싶은        저는       좋아요
0  0.000000  0.000000  0.000000  ...  0.287682  0.000000  0.000000
1  0.000000  0.000000  0.000000  ...  0.287682  0.000000  0.000000
2  0.000000  0.693147  0.693147  ...  0.000000  0.000000  0.000000
3  0.693147  0.000000  0.000000  ...  0.000000  0.693147  0.693147

[4 rows x 9 columns]
'''

from sklearn.feature_extraction.text import CountVectorizer

corpus = [
    'you know I want your love',
    'I like you',
    'what should I do']

vector = CountVectorizer()

print(vector.fit_transform(corpus).toarray())
'''
[[0 1 0 1 0 1 0 1 1]
 [0 0 1 0 0 0 0 1 0]
 [1 0 0 0 1 0 1 0 0]]
'''
print(vector.vocabulary_)
'''
{'you': 7, 'know': 1, 'want': 5, 'your': 8, 'love': 3, 'like': 2, 'what': 6, 'should': 4, 'do': 0}
'''

from sklearn.feature_extraction.text import TfidfVectorizer

# 사이킷런은 TF-IDF를 자동 계산해주는 TfidVectorizer를 제공
corpus = [
    'you know I want your love',
    'I like you',
    'what should I do']

# 1. fit() 로 학습
tfidfv = TfidfVectorizer().fit(corpus)

print(tfidfv.transform(corpus).toarray())
'''
[[0.         0.46735098 0.         0.46735098 0.         0.46735098
  0.         0.35543247 0.46735098]
 [0.         0.         0.79596054 0.         0.         0.
  0.         0.60534851 0.        ]
 [0.57735027 0.         0.         0.         0.57735027 0.
  0.57735027 0.         0.        ]]
'''
print(tfidfv.vocabulary_)
'''
{'you': 7, 'know': 1, 'want': 5, 'your': 8, 'love': 3, 'like': 2, 'what': 6, 'should': 4, 'do': 0}
'''
'''
케라스로도 DTM과 TF-IDF 행렬을 만들 수 있다. => 다중 퍼셉트
'''








































