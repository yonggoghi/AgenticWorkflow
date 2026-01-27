import hashlib

def sha256_hash(text: str) -> str:
    """텍스트의 SHA256 해시값을 반환"""
    return hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]
