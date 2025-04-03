#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 11 14:13:07 2025

@author: oh
"""
'''
비정형 데이터를 분석 후 Word Cloud 생성하기

1. 서울시 응답소 페이지 분석하기  : 서울시 응답소.txt
2. 여고생이 가장 고치고 싶은 성형부위 : 성형상담.txt
3. 성형 수술 부작용 관련 키워드 분석 : 성형부작용.txt

4. 제주도 추천 여행코스 찾기 : 제주도여행지.txt
5. 블로거들이 추천하는 서울 명소 분석하기 : 서울명소.txt

6. 연설문 분석 후 WordCloud 생성하기 : 노무현대통령.txt
7. 대통령 신년 연설문 분석으로 정책 변화 예측하기 : 박근혜대통령취임사_2013.txt
                                        박근혜대통령신년연설문_2014_01_06.txt
                                        박근혜대통령신년연설문_2015_01_12.txt

불용어 사전과 추가 사전은 MySQL로 DB화 (단, 각각의 테이블로 분리 저장)
추가 사전 : 서울명소
불용어 사전 : 박근혜대통령 / 서울명소 / 성형 / 성형부작용 / 제주도여행코스
'''

from soynlp import DoublespaceLineCorpus
from soynlp.word import WordExtractor
from wordcloud import WordCloud
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'AppleGothic'


# ---------------파일 읽기---------------------------------------
import chardet

# 인코딩 감지 함수
def detect_encoding(file_path):
    with open(file_path, "rb") as f:
        raw_data = f.read()
        result = chardet.detect(raw_data)
        return result["encoding"]

# UTF-8 변환 함수
def convert_to_utf8(file_path):
    encoding = detect_encoding(file_path)  # 파일의 인코딩 감지
    with open(file_path, "r", encoding=encoding, errors="ignore") as f:
        text = f.read()
    
    utf8_path = file_path + "_utf8.txt"  # 변환된 파일 저장 경로
    with open(utf8_path, "w", encoding="utf-8") as f:
        f.write(text)
    
    return utf8_path

# 실제 변환 실행
seoul_file_path = "./data files/서울시 응답소.txt"  # 원본 파일 경로
utf8_seoul_file_path = convert_to_utf8(seoul_file_path)  # UTF-8 변환

with open(utf8_seoul_file_path, "r", encoding="utf-8", errors="ignore") as f:
    text = f.read()

# --------------------------------------------------------

corpus = DoublespaceLineCorpus(utf8_seoul_file_path)
print(f"코퍼스 문장 수: {len(corpus)}")

word_extractor = WordExtractor()
word_extractor.train(corpus)
# training was done. used memory 1.066 Gb

word_score_table = word_extractor.extract()
'''
all cohesion probabilities was computed. # words = 226
all branching entropies was computed # words = 506
all accessor variety was computed # words = 506
'''

# 결합도(cohesion_forward)가 높은 순으로 정렬하여 상위 20개 추출
sorted_words = sorted(
    word_score_table.items(), key=lambda x: x[1].cohesion_forward, reverse=True)[:20]

# 주요 단어 출력
for word, score in sorted_words:
    print(f"{word}: {score.cohesion_forward:.4f}")

# -------------------------------------------------

# 워드클라우드 생성
word_freq = {word: score.cohesion_forward for word, score in sorted_words}

wordcloud = WordCloud(
    font_path="AppleGothic",  # macOS: "AppleGothic"
    background_color="white",
    width=800,
    height=400,
).generate_from_frequencies(word_freq)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

# -------------------------------- 꼬꼬마로 다시 --------

from konlpy.tag import Kkma
from collections import Counter

seoul_file_path = "./data files/서울시 응답소.txt"  # 원본 파일 경로
utf8_seoul_file_path = convert_to_utf8(seoul_file_path)  # UTF-8 변환

with open(utf8_seoul_file_path, "r", encoding="utf-8", errors="ignore") as f:
    text = f.read()

kkma = Kkma()

# 형태소 분석 (전체 문장 분석)
morphs = kkma.morphs(text)
print("형태소 분석 결과:", morphs[:50])  # 상위 50개 확인

# 명사만 추출
nouns = kkma.nouns(text)
print("명사 추출 결과:", nouns[:50])  # 상위 50개 확인
'''
명사 추출 결과: ['305', '무료', '무료법률상담', '법률', '상담', '부탁', 
           '말씀', '-27', '2', '304', '교통', '교통불편접수', '불편', 
           '접수', '6715', '버스', '신월', '신월동', '동', '상암', '상암동', 
           '-26', '303', '경기', '경기도', '도', '시흥', '시흥시', '시', '아파트', 
           '화재', '-22', '145', '302', '곡', '곡지구', '지구', '하자', '하자보수', 
           '보수', '관련', '57', '301', '강남', '성수동', '압구정', '압구정역', '역', 
           '방면', '83']
'''

# 워드클라우드 생성
wordcloud = WordCloud(
    font_path="AppleGothic",
    background_color="white",
    width=800,
    height=400
).generate_from_frequencies(word_freq)

# 시각화
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("1. 서울시 응답소 페이지 분석 WordCloud")
plt.show()

# 2 여고생이 가장 고치고 싶은 성형부위

face_file_path = "./data files/성형상담.txt"  # 원본 파일 경로
utf8_face_file_path = convert_to_utf8(face_file_path)  # UTF-8 변환

with open(utf8_face_file_path, "r", encoding="utf-8", errors="ignore") as f:
    text = f.read()

kkma = Kkma()

# 형태소 분석 (전체 문장 분석)
morphs = kkma.morphs(text)
print("형태소 분석 결과:", morphs[:50])  # 상위 50개 확인

# 명사만 추출
nouns = kkma.nouns(text)
print("명사 추출 결과:", nouns[:50])  # 상위 50개 확인
'''
명사 추출 결과: ['커플', '여고생', '여고생쌍', '커플수술', '수술', '쌍', '후부', 
           '후부작용', '작용', '재수술', '의사', '의사답변', '답변', '2012.01', 
           '28', '안녕', '안녕하세', '하세', '대한', '대한의사협회', '협회', '네이버', 
           '지식', '의료', '상담', '성형외과', '전문의', '김', '왕', '문', '주신', '내용', 
           '성형외과전문의', '선생님', '직접', '온라인', '오프라인', '상', '후', '신중', '선택', 
           '건강', '의료상담', '외과', '3', '추', '추천수', '천수', '0', '조']
'''

# 불용어 파일 경로
stopwords_file_path = "./data files/불용어 사전의 예/성형gsub.txt"  # 로컬 불용어 파일

# 불용어 리스트 불러오기
with open(stopwords_file_path, "r", encoding="utf-8", errors="ignore") as f:
    stopwords = set(f.read().splitlines())  # 한 줄씩 읽어서 불용어 리스트 생성
    
# 불용어 제거 적용
filtered_nouns = [word for word in nouns if word not in stopwords]

# 명사 빈도 계산 (불용어 제외)
word_freq = Counter(filtered_nouns)

# 워드클라우드 생성
wordcloud = WordCloud(
    font_path="AppleGothic",
    background_color="white",
    width=800,
    height=400
).generate_from_frequencies(word_freq)

# 시각화
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("2. 여고생이 가장 고치고 싶은 성형부위 WordCloud")
plt.show()

# 3. 성형 수술 부작용 관련 키워드 분석

PLsur_file_path = "./data files/성형부작용.txt"  # 원본 파일 경로
utf8_PLsur_file_path = convert_to_utf8(PLsur_file_path)  # UTF-8 변환

with open(utf8_PLsur_file_path, "r", encoding="utf-8", errors="ignore") as f:
    text = f.read()

kkma = Kkma()

# 형태소 분석 (전체 문장 분석)
morphs = kkma.morphs(text)
print("형태소 분석 결과:", morphs[:50])  # 상위 50개 확인

# 명사만 추출
nouns = kkma.nouns(text)
print("명사 추출 결과:", nouns[:50])  # 상위 50개 확인
'''
명사 추출 결과: ['성형', '성형수술', '수술', '부작용', '의사', '의사답변', '답변', 
           '2015.01', '06', '질', '질성형수술', '대', '보통', '여성', '나이', '출산', 
           '나', '질의', '내부', '탄력', '부부', '부부관계시', '관계', '시', '느낌', '성감', 
           '만족도', '감퇴', '건강', '의료', '의료상담', '상담', '산', '산부인과', '부인과', 
           '4', '추', '추천수', '천수', '0', '조', '조회수', '회수', '1469', '코', '13', 
           '자', '입자', '연골', '연골이식']
'''

# 불용어 파일 경로
stopwords_file_path = "./data files/불용어 사전의 예/성형부작용gsub.txt"  # 로컬 불용어 파일

# 불용어 리스트 불러오기
with open(stopwords_file_path, "r", encoding="utf-8", errors="ignore") as f:
    stopwords = set(f.read().splitlines())  # 한 줄씩 읽어서 불용어 리스트 생성
    
# 불용어 제거 적용
filtered_nouns = [word for word in nouns if word not in stopwords]

# 명사 빈도 계산 (불용어 제외)
word_freq = Counter(filtered_nouns)

# 워드클라우드 생성
wordcloud = WordCloud(
    font_path="AppleGothic",
    background_color="white",
    width=800,
    height=400
).generate_from_frequencies(word_freq)

# 시각화
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("3. 성형 수술 부작용 관련 키워드 분석 WordCloud")
plt.show()

# 4. 제주도 추천 여행코스 찾기

jeju_file_path = "./data files/제주도여행지.txt"  # 원본 파일 경로
utf8_jeju_file_path = convert_to_utf8(jeju_file_path)  # UTF-8 변환

with open(utf8_jeju_file_path, "r", encoding="utf-8", errors="ignore") as f:
    text = f.read()

kkma = Kkma()

# 형태소 분석 (전체 문장 분석)
morphs = kkma.morphs(text)
print("형태소 분석 결과:", morphs[:50])  # 상위 50개 확인

# 명사만 추출
nouns = kkma.nouns(text)
print("명사 추출 결과:", nouns[:50])  # 상위 50개 확인
'''
명사 추출 결과: ['성형', '성형수술', '수술', '부작용', '의사', '의사답변', '답변', 
           '2015.01', '06', '질', '질성형수술', '대', '보통', '여성', '나이', '출산', 
           '나', '질의', '내부', '탄력', '부부', '부부관계시', '관계', '시', '느낌', '성감', 
           '만족도', '감퇴', '건강', '의료', '의료상담', '상담', '산', '산부인과', '부인과', 
           '4', '추', '추천수', '천수', '0', '조', '조회수', '회수', '1469', '코', '13', 
           '자', '입자', '연골', '연골이식']
'''

# 불용어 파일 경로
stopwords_file_path = "./data files/불용어 사전의 예/제주도여행코스gsub.txt"  # 로컬 불용어 파일
utf8_stopwords_file_path = convert_to_utf8(stopwords_file_path)  # UTF-8 변환

# 불용어 리스트 불러오기
with open(utf8_stopwords_file_path, "r", encoding="utf-8", errors="ignore") as f:
    stopwords = set(f.read().splitlines())  # 한 줄씩 읽어서 불용어 리스트 생성
    
# 불용어 제거 적용
filtered_nouns = [word for word in nouns if word not in stopwords]

# 명사 빈도 계산 (불용어 제외)
word_freq = Counter(filtered_nouns)

# 워드클라우드 생성
wordcloud = WordCloud(
    font_path="AppleGothic",
    background_color="white",
    width=800,
    height=400
).generate_from_frequencies(word_freq)

# 시각화
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("4. 제주도 추천 여행코스 찾기 WordCloud")
plt.show()

# 5. 블로거들이 추천하는 서울 명소 분석하

seoul_place_file_path = "./data files/서울명소.txt"  # 원본 파일 경로
utf8_seoul_place_file_path = convert_to_utf8(seoul_place_file_path)  # UTF-8 변환

with open(utf8_seoul_place_file_path, "r", encoding="utf-8", errors="ignore") as f:
    text = f.read()

kkma = Kkma()

# 형태소 분석 (전체 문장 분석)
morphs = kkma.morphs(text)
print("형태소 분석 결과:", morphs[:50])  # 상위 50개 확인

# 명사만 추출
nouns = kkma.nouns(text)
print("명사 추출 결과:", nouns[:50])  # 상위 50개 확인
'''
명사 추출 결과: ['162,900', '162,900건', '건', '서울', '야경', '데이트', '6', '명소', 
           '7', '7일전', '일', '전', '보내기', '달빛', '후원', '모습', '수', '곳', '거', 
           '번째', '북', '북악', '악', '팔각정', '저', '1207', '220147814047', '루', 
           '루디', '디', '반딧불', '세', '로그', '내', '검색', '혼자', '여행', '19',
           '2014.09', '06', '분', '덕수궁', '구경', '방', '방문하시기', '문하', '시기', 
           '옆', '소문', '소문청사']
'''

# 불용어 파일 경로
stopwords_file_path = "./data files/불용어 사전의 예/서울명소gsub.txt"  # 로컬 불용어 파일
utf8_stopwords_file_path = convert_to_utf8(stopwords_file_path)  # UTF-8 변환

# 불용어 리스트 불러오기
with open(utf8_stopwords_file_path, "r", encoding="utf-8", errors="ignore") as f:
    stopwords = set(f.read().splitlines())  # 한 줄씩 읽어서 불용어 리스트 생성
    
# 불용어 제거 적용
filtered_nouns = [word for word in nouns if word not in stopwords]

# 명사 빈도 계산 (불용어 제외)
word_freq = Counter(filtered_nouns)

# 워드클라우드 생성
wordcloud = WordCloud(
    font_path="AppleGothic",
    background_color="white",
    width=800,
    height=400
).generate_from_frequencies(word_freq)

# 시각화
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("5. 블로거들이 추천하는 서울 명소 분석하기 WordCloud")
plt.show()

# 6. 연설문 분석후 wordcloud 생성하기

president_n_file_path = "./data files/노무현대통령.txt"  # 원본 파일 경로
utf8_president_n_file_path = convert_to_utf8(president_n_file_path)  # UTF-8 변환

with open(utf8_president_n_file_path, "r", encoding="utf-8", errors="ignore") as f:
    text = f.read()

kkma = Kkma()

# 형태소 분석 (전체 문장 분석)
morphs = kkma.morphs(text)
print("형태소 분석 결과:", morphs[:50])  # 상위 50개 확인

# 명사만 추출
nouns = kkma.nouns(text)
print("명사 추출 결과:", nouns[:50])  # 상위 50개 확인
'''
명사 추출 결과: ['존경', '국민', '여러분', '사랑', '해외', '해외동포', '동포', 
           '우리', '오늘', '승리', '우리', '승자', '패자', '모두', '대한', 
           '대한민국', '민국', '저', '영광', '새', '역사', '시작', '갈등', 
           '분열', '시대', '7', '7천', '천', '겨레', '하나', '대통', '대통합의', 
           '합의', '원칙', '신뢰', '정치', '평화', '번영', '한반도', '정직', 
           '사람', '성공', '보통', '사회', '투명', '공정', '경제', '노사', '화합', '기업']
'''

# 불용어 파일 경로
stopwords_file_path = "./data files/불용어 사전의 예/서울명소gsub.txt"  # 로컬 불용어 파일
utf8_stopwords_file_path = convert_to_utf8(stopwords_file_path)  # UTF-8 변환

# 불용어 리스트 불러오기
with open(utf8_stopwords_file_path, "r", encoding="utf-8", errors="ignore") as f:
    stopwords = set(f.read().splitlines())  # 한 줄씩 읽어서 불용어 리스트 생성
    
# 불용어 제거 적용
filtered_nouns = [word for word in nouns if word not in stopwords]

# 명사 빈도 계산 (불용어 제외)
word_freq = Counter(filtered_nouns)

# 워드클라우드 생성
wordcloud = WordCloud(
    font_path="AppleGothic",
    background_color="white",
    width=800,
    height=400
).generate_from_frequencies(word_freq)

# 시각화
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("6. 연설문 분석후 wordcloud 생성하기 WordCloud")
plt.show()



# 7. 대통령 신년 연설문 분석으로 정책 변화 예측하기

# 연설문 파일 경로 리스트
speech_files = [
    "./data files/박근혜대통령취임사_2013.txt",
    "./data files/박근혜대통령신년연설문_2015_01_12.txt",
    "./data files/박근혜대통령신년연설문_2014_01_06.txt"
]

# UTF-8 변환된 파일 리스트
utf8_speech_files = [convert_to_utf8(file) for file in speech_files]

# 불용어 파일도 UTF-8 변환
stopwords_file_path = "./data files/불용어 사전의 예/박근혜대통령gsub.txt"
utf8_stopwords_file_path = convert_to_utf8(stopwords_file_path)


# 모든 연설문 텍스트 불러오기
all_text = ""
for file_path in utf8_speech_files:
    with open(file_path, "r", encoding="utf-8") as f:
        all_text += f.read() + "\n"  # 여러 연설문을 하나의 텍스트로 합침

# KKma를 이용한 명사 추출
nouns = kkma.nouns(all_text)
print("명사 추출 결과:", nouns[:50])  # 상위 50개 확인
'''
명사 추출 결과: ['존경', '국민', '분', '700', '해외', '해외동포', 
           '동포', '저', '오늘', '대한', '대한민국', '민국', '제', 
           '제18대', '18', '대', '대통령', '취임', '희망', '시대', 
           '각오', '자리', '시대적', '소명', '주신', '여러분', '감사', 
           '참석', '이명', '이명박', '박', '전직', '세계', '각국', '경축', 
           '경축사절', '사절', '내외', '귀빈', '뜻', '부응', '경제', '경제부흥', 
           '부흥', '국민행복', '행복', '문화', '문화융성', '융성', '부강']
'''


# 불용어 리스트 불러오기
with open(utf8_stopwords_file_path, "r", encoding="utf-8") as f:
    stopwords = set(f.read().splitlines())  # 한 줄씩 불용어 리스트로 변환

print("불용어 리스트 샘플:", list(stopwords)[:20])  # 불용어 상위 20개 확인

# 불용어 제거 적용
filtered_nouns = [word for word in nouns if word not in stopwords]

print("불용어 제거 후 명사 샘플:", filtered_nouns[:50])

    
# 불용어 제거 적용
filtered_nouns = [word for word in nouns if word not in stopwords]

# 명사 빈도 계산 (불용어 제외)
word_freq = Counter(filtered_nouns)

# 워드클라우드 생성
wordcloud = WordCloud(
    font_path="AppleGothic",
    background_color="white",
    width=800,
    height=400
).generate_from_frequencies(word_freq)

# 시각화
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("7. 대통령 신년 연설문 분석으로 정책 변화 예측하기 WordCloud")
plt.show()












