# app_chatbot/utils/file_utils.py

import os
import glob
import json
from typing import Optional, Dict, Any


ROOT_DATASET_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "dataset")
USER_INFO_DIR = os.path.join(ROOT_DATASET_DIR, "user_info")
CHAT_HISTORY_DIR = os.path.join(ROOT_DATASET_DIR, "chat_histories")


def ensure_directory(path: str):
    '''
    디렉토리가 존재하지 않을 경우 생성
    '''
    os.makedirs(path, exist_ok=True)


def load_json(file_path: str) -> Optional[Dict[str, Any]]:
    '''
    JSON 파일 불러오는 함수
    '''
    if not os.path.exists(file_path):
        # 파일이 없으면 None 반환
        # 호출자에서 None 체크하도록 설계
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
    JSON 저장하는 함수
    '''
    try:
        dirpath = os.path.dirname(file_path)
        if dirpath:
            ensure_directory(dirpath)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        return True
    except Exception as e:
        print(f"JSON 저장 중 오류 발생: {e}")
        return False


def _user_info_path(user_id: str) -> str:
    '''
    user_info 파일 경로 반환
    '''
    ensure_directory(USER_INFO_DIR)
    return os.path.join(USER_INFO_DIR, f"{user_id}.json")


def load_user_info(user_id: str) -> Optional[Dict[str, Any]]:
    '''
    user_info/{user_id}.json 파일을 불러오는 헬퍼
    '''
    path = _user_info_path(user_id)
    return load_json(path)


def save_user_info(user_id: str, data: Dict[str, Any]) -> bool:
    '''
    user_info/{user_id}.json 파일로 저장하는 헬퍼
    '''
    path = _user_info_path(user_id)
    return save_json(path, data)


def _chat_history_dir_for_user(user_id: str) -> str:
    '''
    사용자별 chat_histories 디렉토리 경로 반환
    유지보수 편하게 user_id 별로 관리..
    '''
    path = os.path.join(CHAT_HISTORY_DIR, user_id)
    ensure_directory(path)
    return path


def chat_history_path(user_id: str, chat_id: str) -> str:
    '''
    특정 user_id, chat_id에 해당하는 json 파일 경로 반환
    '''
    dirpath = _chat_history_dir_for_user(user_id)
    return os.path.join(dirpath, f"{chat_id}.json")
