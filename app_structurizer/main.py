# app_structurizer/main.py

import os
import time
import json
import traceback
from dotenv import load_dotenv
from structurizer_utils import extract_structured_idea_info
from sqs_server import (
    get_sqs_resource,
    get_queue,
    has_messages_in_queue,
    receive_message_from_sqs,
    send_message_to_sqs,
    delete_message_from_sqs,
)

# 환경 변수 로드
load_dotenv()


def main():
    """SQS로부터 메시지를 받아 구조화 후 응답 큐로 전송하는 메인 프로세스"""
    # 환경 변수 불러오기
    request_queue_url = os.getenv("AWS_SQS_METADATA_REQUEST_QUEUE")
    response_queue_url = os.getenv("AWS_SQS_METADATA_RESPONSE_QUEUE")
    region_name = os.getenv("AWS_DEFAULT_REGION", "ap-northeast-2")

    print(f"\n✅ 환경변수 로드 완료:")
    print(f"   - Request Queue: {request_queue_url}")
    print(f"   - Response Queue: {response_queue_url}")
    print(f"   - Region: {region_name}")

    # SQS 리소스 및 큐 객체 생성
    sqs = get_sqs_resource(region_name)
    request_queue = get_queue(sqs, request_queue_url)
    response_queue = get_queue(sqs, response_queue_url)

    print("\n🔄 SQS 메시지 처리 루프 시작...\n")

    processed_count = 0

    # 메시지 처리 루프
    while has_messages_in_queue(request_queue):
        try:
            processed_count += 1
            print(f"\n{'=' * 60}")
            print(f"📨 메시지 #{processed_count} 처리 중...")
            print(f"{'=' * 60}\n")

            # SQS로부터 데이터 가져오기
            body, message = receive_message_from_sqs(request_queue)
            if not body:
                print("⚠️ 메시지 없음 또는 파싱 실패, 다음 메시지로 이동")
                continue

            query_text = body.get("query") or body.get("text") or ""
            print(f"🚀 파이프라인 실행: {query_text[:80]}...")

            # 아이디어 구조화 실행
            result = extract_structured_idea_info(query_text)

            # 응답 큐로 결과 전송
            print(f"📤 결과를 SQS 응답 큐로 전송 중...")
            message_id = send_message_to_sqs(
                response_queue,
                result,
                message_group_id="metadata-group"
            )

            if message_id:
                print(f"✅ 메시지 #{processed_count} 처리 완료")
                print(f"   MessageId: {message_id}")
            else:
                print(f"❌ 메시지 #{processed_count} 결과 전송 실패")

            # 메시지 삭제 (중복 처리 방지)
            delete_message_from_sqs(message)
            print("🗑️ 원본 메시지 삭제 완료")

        except Exception as e:
            print(f"\n❌ 메시지 #{processed_count} 처리 중 오류 발생:")
            print(f"   {type(e).__name__}: {e}")
            print(traceback.format_exc())
            # 오류 발생 시 다음 메시지로 계속
            continue

        # 과도한 API 호출 방지를 위한 잠시 대기
        time.sleep(1)

    print(f"\n{'=' * 60}")
    print(f"✅ 전체 처리 완료 — 총 {processed_count}개 메시지 처리")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()