import hashlib
import hmac
import time
from dataclasses import dataclass

@dataclass
class V2XMessage:
    sender_id: str
    payload: str
    timestamp: float
    signature: str

class V2XCryptoProvider:
    """
    Simulates IEEE 1609.2 PKI Digital Signature generation and verification for V2X safety broadcasts.
    """
    def __init__(self, secret_key: bytes = b"V2X_SECURE_AUTONOMOUS_KEY_2026"):
        self.secret_key = secret_key

    def sign_message(self, sender_id: str, payload: str) -> V2XMessage:
        now = time.time()
        msg_str = f"{sender_id}:{payload}:{now}"
        signature = hmac.new(self.secret_key, msg_str.encode('utf-8'), hashlib.sha256).hexdigest()
        return V2XMessage(
            sender_id=sender_id,
            payload=payload,
            timestamp=now,
            signature=signature
        )

    def verify_message(self, message: V2XMessage) -> bool:
        msg_str = f"{message.sender_id}:{message.payload}:{message.timestamp}"
        expected_sig = hmac.new(self.secret_key, msg_str.encode('utf-8'), hashlib.sha256).hexdigest()
        # Verify MAC match and freshness (timestamp within 5.0 seconds)
        fresh = (time.time() - message.timestamp) < 5.0
        valid_sig = hmac.compare_digest(expected_sig, message.signature)
        return fresh and valid_sig
