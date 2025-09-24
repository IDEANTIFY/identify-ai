AI 기반 아이디어 검증 및 리포팅 API
본 프로젝트는 사용자의 아이디어를 입력받아 유효성을 검증하고, 상세 분석 리포트를 생성하는 AI 기반 API 서버입니다.

프로젝트는 두 개의 마이크로서비스(MSA)로 구성되어 있습니다:

아이디어 구조화 API: 사용자의 아이디어 텍스트를 구조화된 JSON 데이터로 변환합니다.

아이디어 검증 API: 구조화된 JSON 데이터를 기반으로 웹/내부 DB 검색을 수행하고 최종 분석 리포트를 생성합니다.

📂 프로젝트 구조
.
├── app_structurizer/
│   ├── __init__.py
│   ├── main.py                 # App 1: 아이디어 구조화 FastAPI 서버
│   └── structurizer_utils.py   # 구조화 정보 추출 로직
│
├── app_validator/
│   ├── __init__.py
│   ├── main.py                 # App 2: 아이디어 검증 FastAPI 서버
│   ├── pipeline.py             # 메인 검증 파이프라인 로직
│   └── utils/                  # 검증 로직 관련 유틸리티 모음
│       ├── __init__.py
│       ├── create_report.py
│       ├── db_search_utils.py
│       ├── query_utils.py
│       └── web_search_utils.py
│
├── dataset/                    # FAISS 인덱싱을 위한 데이터셋
├── models/                     # Sentence Transformer 모델
└── requirements.txt            # Python 의존성 패키지 목록

🚀 시작하기
사전 준비사항
Python (3.11 이상 권장)

Docker Desktop (Docker 실행 시 필요)

로컬 환경에서 실행하기
프로젝트 경로로 이동

cd conference_2025

가상환경 생성 및 활성화

# 가상환경 생성
python -m venv env

# 가상환경 활성화 (macOS/Linux)
source env/bin/activate

필요 패키지 설치

pip install -r requirements.txt

API 서버 실행

💡 참고: 각 API 서버는 별도의 터미널에서 실행해야 합니다.

App 1: 아이디어 구조화 서버 실행 (Port: 8001)

uvicorn app_structurizer.main:app --reload --port 8001

App 2: 아이디어 검증 서버 실행 (Port: 8002)

uvicorn app_validator.main:app --reload --port 8002

🐳 Docker로 실행하기
⚠️ 참고: 각 애플리케이션의 루트 디렉토리(app_structurizer/, app_validator/)에 Dockerfile이 존재해야 합니다.

Docker 이미지 빌드

App 1: 아이디어 구조화 이미지

docker build -t structurizer-api:1.0 -f app_structurizer/Dockerfile .

App 2: 아이디어 검증 이미지

docker build -t validator-api:1.0 -f app_validator/Dockerfile .

Docker 컨테이너 실행

App 1: 아이디어 구조화 컨테이너

docker run -d -p 8001:8001 --name structurizer-container structurizer-api:1.0

App 2: 아이디어 검증 컨테이너

docker run -d -p 8002:8002 --name validator-container validator-api:1.0

📜 API 명세 (API Specification)
각 서버가 실행되면, 브라우저에서 /docs 경로로 접속하여 API를 직접 테스트해볼 수 있는 Swagger UI를 확인할 수 있습니다.

App 1: 아이디어 구조화 API
Swagger UI: http://localhost:8001/docs

POST /structure-idea/
사용자의 아이디어 텍스트를 받아 구조화된 JSON을 반환합니다.

Request Body:

{
  "idea_text": "AI를 활용해서 개인 맞춤형 여행 코스를 짜주는 앱"
}

Response Body:

{
  "주요_내용": "AI 기반 맞춤형 여행 코스 추천",
  "도메인": "여행, AI",
  "목적": "개인화된 여행 경험 제공",
  "차별성": "실시간 교통 정보와 사용자 리뷰를 종합하여 동적으로 코스 변경",
  "핵심_기술": "자연어 처리, 추천 알고리즘",
  "서비스_대상": "자유여행을 선호하는 20-30대"
}

App 2: 아이디어 검증 API
Swagger UI: http://localhost:8002/docs

POST /validate-idea/
구조화된 아이디어 JSON을 받아 최종 검증 리포트를 반환합니다.

Request Body:

{
  "주요_내용": "AI 기반 맞춤형 여행 코스 추천",
  "도메인": "여행, AI",
  "목적": "개인화된 여행 경험 제공",
  "차별성": "실시간 교통 정보와 사용자 리뷰를 종합하여 동적으로 코스 변경",
  "핵심_기술": "자연어 처리, 추천 알고리즘",
  "서비스_대상": "자유여행을 선호하는 20-30대"
}

Response Body:

{
  "summary_report": {
    "assessment": "...",
    "similar_cases": "..."
  },
  "detailed_report": {
    "web_sources": [
      { "title": "...", "url": "...", "summary": "..." }
    ],
    "db_sources": [
      { "title": "...", "similarity": 0.85, "summary": "..." }
    ]
  }
}
