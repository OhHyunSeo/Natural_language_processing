#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 11 09:20:21 2025

@author: oh
"""

import urllib.request
from soynlp import DoublespaceLineCorpus
from soynlp.word import WordExtractor

urllib.request.urlretrieve("https://raw.githubusercontent.com/lovit/soynlp/master/tutorials/2016-10-20.txt",
filename="2016-10-20.txt")

# 훈련 데이터를 다수의 문서로 분리

corpus = DoublespaceLineCorpus("2016-10-20.txt")
len(corpus) # Out[4]: 30091

# 상위 3개의 문서만 출력
i = 0
for document in corpus:
    if len(document) > 0:
        print(document)
        
        i = i + 1
        
    if i == 3:
        break
    
'''
soynlp는 학습 기반의 단어 토크나이저이므로
기존의 KoNLPy에서 제공하는 형태소 분석기들과는 달리 학습 과정을 거쳐야 한다.
=> 전체 코퍼스로부터 응집 확률과 브랜칭 엔트로피 단어 점수표를 만드는 과정.
=> WordExtractor.extract()를 통해서 전체 코퍼스에 대해 단어 점수표를 계산.

응집 확률 : 내부 문자열에 얼마나 응집하여 자주 등장하는지를 판단하는 척도
-> 문자열을 문자 단위로 분리하여 내부 문자열을 만드는 과정에서
-> 왼쪽부터 순서대로 문자를 추가하면서 각 문자열이 주어졌을때 그 다음 문자가 나올 확률을 계산하여 누적곱을 한 값
=> 이 값이 높을 수록 전체 코퍼스에서 이 문자열 시퀀스는 하나의 단어로 등장할 가능성이 높다.
'''
word_extractor = WordExtractor()
word_extractor.train(corpus)
# training was done. used memory 1.066 Gb

word_score_table = word_extractor.extract()
'''
all cohesion probabilities was computed. # words = 223348
all branching entropies was computed # words = 361598
all accessor variety was computed # words = 361598
'''

# '반포한'의 응집 확률 계산
word_score_table["반포한"].cohesion_forward
# Out[8]: 0.08838002913645132

# '반포한강'의 응집 확률 계산
word_score_table["반포한강"].cohesion_forward
# Out[9]: 0.19841268168224552

# '반포한강공'의 응집 확률 계산
word_score_table["반포한강공"].cohesion_forward
# Out[10]: 0.2972877884078849

# '반포한강공원'의 응집 확률 계산
word_score_table["반포한강공원"].cohesion_forward
# Out[11]: 0.37891487632839754

# '반포한강공원에'의 응집 확률 계산
word_score_table["반포한강공원에"].cohesion_forward
# Out[12]: 0.33492963377557666

# 결합도는 '반포한강공원'일 때가 가장높다
# 응집도를 통해 판단하기에 하나의 단어로 판단하기에 가장 적합한 문자열은 '반포한강공원'

'''
SOYNLP의 브랜칭 엔트로피(branching entropy)
Branching Entropy는 확률 분포의 엔트로피값을 사용.
=> 주어진 문자열에서 얼마나 다음 문자가 등장할 수 있는지를 판단하는 척도

브랜칭 엔트로피의 값은
하나의 완성된 단어에 가까워질수록 문맥으로 인해 점점 정확히 예측할 수 있게 되면서 점점 줄어든다.

첫번째 문지는 '디'
정답은 '스'

'디스' 다음 문자는
정답은 '플'
정답은 '레'
정답은 '이'
'''

word_score_table["디스"].right_branching_entropy # Out[13]: 1.6371694761537934
word_score_table["디스플"].right_branching_entropy # Out[14]: -0.0
word_score_table["디스플레"].right_branching_entropy # Out[15]: -0.0
word_score_table["디스플레이"].right_branching_entropy # Out[17]: 3.1400392861792916

'''
하나의 단어가 끝나면 그 경계 부분부터 다시 브랜칭 엔트로피 값이 증가하게 됨 => 단어를 판단하는 것이 가능
'''

'''
SOYNLP의 L tokenizer

한국어는 띄어쓰기 단위로 나눈 어절 토큰은 주로 L 토큰 + R 토큰의 형식을 가질 때가 많다.
예) '공원에'는 '공원 + 에'로 나눌 수 있다. 또는 '공부하는'은 '공부 + 하는'으로 나눌 수도 있을 것이다.
    
L 토크나이저는
L 토큰 + R 토큰으로 나누되,
분리 기준을 점수가 가장 높은 L 토큰을 찾아내는 원리를 가지고 있다.

'''
from soynlp.tokenizer import LTokenizer
'''
word_score_table => word, score
score.cohension_forward
Ltokenizer(scores = scores)
'''

scores = {word:score.cohesion_forward for word, score in word_score_table.items()}
'''
 '돗': 0,
 '똑': 0,
 '허': 0,
 '냈': 0,
 '악': 0,
 ...}
'''

l_tokenizer = LTokenizer(scores = scores)
l_tokenizer.tokenize('국제사회와 우리의 노력들로 범죄를 척결하자', flatten = False)
'''
Out[22]: [('국제사회', '와'), ('우리', '의'), ('노력', '들로'), ('범죄', '를'), ('척결', '하자')]
'''
'''
띄어쓰기가 되어 있지 않은 문장을 넣어서 점수를 통해 토큰화 된 결과를 확인.

'''
from soynlp.tokenizer import MaxScoreTokenizer

maxscore_tokenizer = MaxScoreTokenizer(scores=scores)
maxscore_tokenizer.tokenize("국제사회와우리의노력들로범죄를척결하자")
'''
Out[25]: ['국제사회', '와', '우리', '의', '노력', '들로', '범죄', '를', '척결', '하자']
'''
'''
SOYNLP를 이용한 반복되는 문자 정제

SNS나 채팅 데이터와 같은 한국어 데이터의 경우에는 ㅋㅋ, ㅎㅎ 등의 이모티콘의 경우 불필요하게 연속되는 경우가 많은데
ㅋㅋ, ㅋㅋㅋ, ㅋㅋㅋㅋ와 같은 경우를 모두 서로 다른 단어로 처리하는 것은 불필요.
=> 반복되는 것은 하나로 정규화시켜준다
'''

from soynlp.normalizer import *


# 앜ㅋㅋㅋㅋ이영화존잼쓰ㅠㅠㅠㅠㅠ
print(emoticon_normalize('앜ㅋㅋㅋㅋ이영화존잼쓰ㅠㅠㅠㅠㅠ', num_repeats=2))
print(emoticon_normalize('앜ㅋㅋㅋㅋㅋㅋㅋㅋㅋ이영화존잼쓰ㅠㅠㅠㅠ', num_repeats=2))
print(emoticon_normalize('앜ㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ이영화존잼쓰ㅠㅠㅠㅠㅠㅠ', num_repeats=2))
print(emoticon_normalize('앜ㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ이영화존잼쓰ㅠㅠㅠㅠㅠㅠㅠㅠ', num_repeats=2))

# 아ㅋㅋ영화존잼쓰ㅠㅠ

'''
의미없게 반복되는 것은 비단 이모티콘에 한정되지 않는다.

'''
print(repeat_normalize('와하하하하하하하하하핫', num_repeats=2))
print(repeat_normalize('와하하하하하하핫', num_repeats=2))
print(repeat_normalize('와하하하하핫', num_repeats=2))
# 와하하핫

'''
Customized KoNLPy

형태소 분석 입력 : '은경이는 사무실로 갔습니다.'
형태소 분석 결과 : ['은', '경이', '는', '사무실', '로', '갔습니다.']

pip install customized_konlpy
'''
from ckonlpy.tag import Twitter

twitter = Twitter()
twitter.morphs('은경이는 사무실로 갔습니다.')
# Out[35]: ['은', '경이', '는', '사무실', '로', '갔습니다', '.']

# 형태소 분석기 Twitter에 add_dictionary('단어', '품사')와 같은 형식으로 사전 추가
twitter.add_dictionary('은경이', 'Noun')
twitter.morphs('은경이는 사무실로 갔습니다.')
# Out[39]: ['은경이', '는', '사무실', '로', '갔습니다', '.']


'''

'''


































