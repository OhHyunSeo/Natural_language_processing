# 🧠 Natural Language Processing (Korean NLP)

[![Python Version](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![KoNLPy](https://img.shields.io/badge/NLP-KoNLPy-green.svg)](https://konlpy.org/)

## 🔍 소개

이 프로젝트는 한국어 자연어 처리를 실습하고 다양한 형태소 분석기 및 토크나이저를 테스트하는 실험용 코드 모음입니다.  
KoNLPy와 Soynlp 등의 라이브러리를 활용하여 전처리, 토큰화, Bag-of-Words 기반 분석 등을 수행합니다.

---

## 🧩 주요 기능

- 📄 텍스트 파일을 불러와 KoNLPy 형태소 분석기(BOW)로 분석
- 🧪 Soynlp를 활용한 어절 기반 분리 테스트
- 🧹 워드 전처리 및 실험용 통합 스크립트 제공

---

## 📁 프로젝트 구조

```
📁 Natural_language_processing-master/
│
├── 2016-10-20.txt       # 분석 대상 원문 데이터 (예: 뉴스 기사)
├── Konlpy_Bow_use.py    # KoNLPy 형태소 분석 + BOW 벡터화
├── soynlp_test.py       # soynlp 기반 토큰 분리 실험
└── work.py              # 실험 통합 실행 스크립트
```

---

## 🚀 실행 방법

### 1. 환경 설정

```bash
python -m venv venv
source venv/bin/activate
pip install konlpy soynlp
```

> Mac 사용자는 Java 설치 필요: `brew install openjdk`

### 2. 실행 예시

```bash
python Konlpy_Bow_use.py     # KoNLPy 분석 실행
python soynlp_test.py        # Soynlp 분석 실행
python work.py               # 통합 전처리/분석 실행
```

---

## 📈 활용 예시

- 키워드 추출
- 빈도 기반 감성 분석
- 형태소 분석 비교 실험

---

## 🧑‍💻 기여 방법

1. 이 저장소를 포크하세요.
2. 새로운 브랜치를 만드세요: `git checkout -b feature/새기능`
3. 변경사항을 커밋하세요: `git commit -m "Add 새기능"`
4. 브랜치에 푸시하세요: `git push origin feature/새기능`
5. Pull Request를 생성하세요.

