import hashlib
import hmac
import json


# ==============================================================
# DEMONSTRATION KEY
# ==============================================================
#
# This is intentionally a local simulation key.
#
# In a real vehicle/security deployment, secrets should NOT be
# hard-coded inside application source code.
# ==============================================================

DEFAULT_DEMO_KEY = b"autonomous-vehicle-research-demo-key-v1"


# ==============================================================
# CANONICAL PAYLOAD
# ==============================================================

def canonical_payload(payload):

    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


# ==============================================================
# SIGN PAYLOAD
# ==============================================================

def sign_payload(
    payload,
    key=DEFAULT_DEMO_KEY,
):

    message = canonical_payload(
        payload
    )

    signature = hmac.new(
        key,
        message,
        hashlib.sha256,
    ).hexdigest()

    return signature


# ==============================================================
# VERIFY PAYLOAD
# ==============================================================

def verify_payload(
    payload,
    signature,
    key=DEFAULT_DEMO_KEY,
):

    expected_signature = sign_payload(
        payload,
        key,
    )

    return hmac.compare_digest(
        expected_signature,
        str(signature),
    )


# ==============================================================
# CREATE SIGNED MESSAGE
# ==============================================================

def make_signed_message(
    payload,
    key=DEFAULT_DEMO_KEY,
):

    return {
        "payload": payload,
        "signature": sign_payload(
            payload,
            key,
        ),
    }


# ==============================================================
# VERIFY SIGNED MESSAGE
# ==============================================================

def verify_signed_message(
    message,
    key=DEFAULT_DEMO_KEY,
):

    if not isinstance(
        message,
        dict,
    ):
        return False

    if "payload" not in message:
        return False

    if "signature" not in message:
        return False

    return verify_payload(
        message["payload"],
        message["signature"],
        key,
    )