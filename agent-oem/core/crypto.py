import base64
from Crypto.Cipher import AES
from Crypto.Protocol.KDF import PBKDF2
from Crypto.Random import get_random_bytes
import hashlib

SALT_LEN = 16
NONCE_LEN = 12
TAG_LEN = 16
ITERATIONS = 100000
HASH_ALGO = hashlib.sha256

def derive_key(password: str, salt: bytes) -> bytes:
    """Derive a 32‑byte AES key from password and salt using PBKDF2."""
    return PBKDF2(password, salt, dkLen=32, count=ITERATIONS, hmac_hash_module=HASH_ALGO)

def encrypt_message(password: str, plaintext: str) -> str:
    """
    Encrypt a plaintext message with AES‑GCM.
    Returns base64‑encoded string of: salt (16) + nonce (12) + tag (16) + ciphertext.
    """
    salt = get_random_bytes(SALT_LEN)
    nonce = get_random_bytes(NONCE_LEN)
    key = derive_key(password, salt)

    cipher = AES.new(key, AES.MODE_GCM, nonce=nonce)
    ciphertext, tag = cipher.encrypt_and_digest(plaintext.encode('utf-8'))

    # Concatenate: salt + nonce + tag + ciphertext
    raw = salt + nonce + tag + ciphertext
    return base64.b64encode(raw).decode('ascii')

def decrypt_message(password: str, encoded: str) -> str:
    """
    Decrypt a base64‑encoded message produced by encrypt_message().
    Returns the original plaintext.
    """
    raw = base64.b64decode(encoded)
    if len(raw) < SALT_LEN + NONCE_LEN + TAG_LEN:
        raise ValueError("Encoded data too short")

    salt = raw[:SALT_LEN]
    nonce = raw[SALT_LEN:SALT_LEN + NONCE_LEN]
    tag = raw[SALT_LEN + NONCE_LEN:SALT_LEN + NONCE_LEN + TAG_LEN]
    ciphertext = raw[SALT_LEN + NONCE_LEN + TAG_LEN:]

    key = derive_key(password, salt)
    cipher = AES.new(key, AES.MODE_GCM, nonce=nonce)

    try:
        plaintext = cipher.decrypt_and_verify(ciphertext, tag)
    except ValueError as e:
        raise ValueError("Decryption failed – wrong password or corrupted data") from e

    return plaintext.decode('utf-8')
