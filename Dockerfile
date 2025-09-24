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
# app, models, dataset 폴더를 모두 컨테이너 안으로 복사합니다.
COPY ./app /code/app
COPY ./models /code/models
COPY ./dataset /code/dataset

# 6. API 서버 실행
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]