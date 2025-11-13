# 1. 베이스 이미지 선택
FROM python:3.11-slim

# 2. 작업 디렉토리 설정
WORKDIR /code

# 3. 환경 변수 설정
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# 4. 의존성 설치 (빌드 캐시 활용)
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade pip -r /code/requirements.txt

# 5. 프로젝트 파일 복사 (★★★★★ 중요 ★★★★★)
COPY ./app /code/app
COPY ./app_chatbot /code/app_chatbot
COPY ./app_issue /code/app_issue
COPY ./app_keyword /code/app_keyword
COPY ./app_structurizer /code/app_structurizer
COPY ./app_validator /code/app_validator

# dataset 복사
COPY ./dataset /code/dataset
# 6. API 서버 실행
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]