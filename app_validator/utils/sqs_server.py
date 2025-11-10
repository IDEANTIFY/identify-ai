"""
SQS 관련 유틸리티 함수 모듈 (boto3.resource 버전)
"""
import os
import json
import time
import boto3
from botocore.exceptions import ClientError


def get_sqs_resource(region_name='ap-northeast-2'):
    """
    SQS 리소스 생성
    
    Args:
        region_name: AWS 리전명 (기본값: ap-northeast-2)
    
    Returns:
        boto3 SQS 리소스 객체
    """
    aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    return boto3.resource(
        'sqs',
        region_name=region_name,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key
    )


def get_queue(sqs_resource, queue_url: str):
    """
    큐 URL로부터 Queue 객체 반환
    """
    return sqs_resource.Queue(queue_url)


def has_messages_in_queue(queue) -> bool:
    """
    큐에 메시지가 있는지 확인
    
    Args:
        queue: boto3 SQS Queue 객체
    
    Returns:
        bool: 메시지가 있으면 True, 없으면 False
    """
    try:
        attributes = queue.attributes
        count = int(attributes.get('ApproximateNumberOfMessages', 0))
        return count > 0
    except Exception as e:
        print(f"큐 확인 오류: {e}")
        return False


def get_queue_message_count(queue) -> int:
    """
    큐의 메시지 개수 조회
    
    Args:
        queue: boto3 SQS Queue 객체
    
    Returns:
        int: 메시지 개수, 오류 시 -1
    """
    try:
        attributes = queue.attributes
        return int(attributes.get('ApproximateNumberOfMessages', 0))
    except Exception as e:
        print(f"메시지 개수 조회 오류: {e}")
        return -1


def receive_message_from_sqs(queue, wait_time: int = 20):
    """
    SQS에서 메시지 하나 수신 (Long Polling)
    
    Args:
        queue: boto3 SQS Queue 객체
        wait_time: Long Polling 대기 시간 (초, 기본값: 20)
    
    Returns:
        tuple: (메시지 본문 dict, message 객체) 또는 (None, None)
    """
    try:
        messages = queue.receive_messages(
            MaxNumberOfMessages=1,
            WaitTimeSeconds=wait_time,
            MessageAttributeNames=['All']
        )

        if messages:
            message = messages[0]
            body = json.loads(message.body)
            return body, message
        return None, None

    except json.JSONDecodeError as e:
        print(f"JSON 파싱 오류: {e}")
        return None, None
    except ClientError as e:
        print(f"메시지 수신 오류: {e}")
        return None, None
    except Exception as e:
        print(f"예상치 못한 수신 오류: {e}")
        return None, None


def receive_messages_batch(queue, max_messages: int = 10, wait_time: int = 20):
    """
    SQS에서 여러 메시지 수신 (배치)
    
    Args:
        queue: boto3 SQS Queue 객체
        max_messages: 한 번에 받을 최대 메시지 수 (1~10)
        wait_time: Long Polling 대기 시간 (초)
    
    Returns:
        list[tuple[dict, Message]]: [(메시지 내용 dict, message 객체), ...]
    """
    try:
        messages = queue.receive_messages(
            MaxNumberOfMessages=max_messages,
            WaitTimeSeconds=wait_time,
            MessageAttributeNames=['All']
        )

        result = []
        for msg in messages:
            try:
                body = json.loads(msg.body)
                result.append((body, msg))
            except json.JSONDecodeError:
                print(f"메시지 파싱 실패: {msg.body[:100]}")
                continue
        return result

    except ClientError as e:
        print(f"배치 수신 오류: {e}")
        return []
    except Exception as e:
        print(f"예상치 못한 배치 수신 오류: {e}")
        return []


def delete_message_from_sqs(message):
    """
    수신한 메시지를 큐에서 삭제
    
    Args:
        message: boto3 SQS Message 객체
    
    Returns:
        bool: 성공 시 True, 실패 시 False
    """
    try:
        message.delete()
        return True
    except ClientError as e:
        print(f"메시지 삭제 오류: {e}")
        return False
    except Exception as e:
        print(f"예상치 못한 삭제 오류: {e}")
        return False


def send_message_to_sqs(
    queue,
    message_body: dict,
    message_group_id: str = "idea-report-group",
    message_attributes: dict = None
) -> str | None:
    """
    FIFO 큐 전용: SQS로 메시지 전송

    Args:
        queue: boto3 SQS Queue 리소스 객체
        message_body: 전송할 메시지 본문 (dict)
        message_group_id: FIFO 큐용 그룹 ID (기본값: "idea-report-group")
        message_attributes: 선택적 메시지 속성

    Returns:
        str | None: 성공 시 MessageId, 실패 시 None
    """
    if message_attributes is None:
        message_attributes = {}

    try:
        # FIFO 큐 전용 파라미터 구성
        dedup_id = f"{message_group_id}-{int(time.time() * 1000)}"

        params = {
            "MessageBody": json.dumps(message_body, ensure_ascii=False),
            "MessageAttributes": message_attributes,
            "MessageGroupId": message_group_id,
            "MessageDeduplicationId": dedup_id
        }

        response = queue.send_message(**params)

        print("[SQS FIFO 전송 요청 파라미터]")
        print(json.dumps(params, ensure_ascii=False, indent=2))
        print("\n[SQS FIFO 응답]")
        print(json.dumps(response, ensure_ascii=False, indent=2))

        msg_id = response.get("MessageId")

        if not msg_id:
            print("MessageId 없음 — 전송 실패로 간주")
            return None

        print(f"MessageId: {msg_id}")
        print(f"전송 대상 ARN: {queue.attributes.get('QueueArn', 'N/A')}")
        return msg_id

    except ClientError as e:
        print(f"SQS 메시지 전송 오류: {e}")
        return None
    except Exception as e:
        print(f"예기치 못한 전송 오류: {e}")
        return None
