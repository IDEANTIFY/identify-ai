# app_chatbot/utils/file_utils.py

import os
import glob
import json
from typing import Optional, Dict, Any


def load_json(file_path: str) -> Optional[Dict[str, Any]]:
    '''
    JSON 파일을 안전하게 불러오는 함수
    '''
    # 파일 존재 여부 확인
    if not os.path.exists(file_path):
        print(f"JSON 파일을 찾을 수 없음: {file_path}")
        return None
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        print(f"JSON 파일 파싱 오류 발생: {file_path}")
        return None
    except Exception as e:
        print(f"JSON 파일 읽기 오류 발생: {e}")
        return None


def save_json(file_path: str, data: Dict[str, Any]) -> bool:
    '''
    JSON 데이터를 파일로 저장하는 함수
    '''
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        return True
    except Exception as e:
        print(f"JSON 저장 중 오류 발생: {e}")
        return False


def find_latest_file(directory: str, prefix: str = "") -> Optional[str]:
    '''
    지정된 디렉토리에서 prefix로 시작하는 최신 파일을 찾는 함수
    '''
    pattern = os.path.join(directory, f"{prefix}*")
    files = glob.glob(pattern)
    if not files:
        return None
    return max(files, key=os.path.getctime)


def ensure_directory(path: str):
    '''
    디렉토리가 존재하지 않을 경우 생성
    '''
    os.makedirs(path, exist_ok=True)
