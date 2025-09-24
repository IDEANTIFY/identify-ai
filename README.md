# identify-ai


- 구조
├── app_structurizer/
│   ├── __init__.py
│   ├── main.py                 # (신규) App 1의 FastAPI 서버
│   └── structurizer_utils.py   # extract_structured_idea_info 함수만 포함
│
├── app_validator/
│   ├── __init__.py
│   ├── main.py                 # (신규) App 2의 FastAPI 서버
│   └── pipeline.py             # 기존 run_idea_validation_pipeline.py의 수정본
│   └── utils/                  # App 2의 유틸리티 함수 모음
│       ├── __init__.py
│       ├── create_report.py
│       ├── db_search_utils.py
│       ├── query_utils.py      # generate_search_query 함수 포함
│       └── web_search_utils.py
│
├── dataset/
├── models/
└── requirements.txt
