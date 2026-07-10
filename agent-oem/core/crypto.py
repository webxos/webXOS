import base64
from Crypto.Cipher import AES
from Crypto.Protocol.KDF import PBKDF2
from Crypto.Random import get_random_bytes
import hashlib

async def derive_key(password: str, salt: bytes):
    key = PBKDF2(password, salt, dkLen=32, count=100000, hmac_hash_module=hashlib.sha256)
    return key

async def encrypt_message(password: str, plaintext: str) -> str:
    salt = get_random_bytes(16)
    iv = get_random_bytes(12)
    key = await derive_key(password, salt)
    cipher = AES.new(key, AES.MODE_GCM, nonce=iv)
    ciphertext, tag = cipher.encrypt_and_digest(plaintext.encode())
    combined = salt + iv + tag + ciphertext
    return base64.b64encode(combined).decode()

async def decrypt_message(password: str, encoded: str) -> str:
    data = base64.b64decode(encoded)
    salt, iv, tag, ciphertext = data[:16], data[16:28], data[28:44], data[44:]
    key = await derive_key(password, salt)
    cipher = AES.new(key, AES.MODE_GCM, nonce=iv)
    plaintext = cipher.decrypt_and_verify(ciphertext, tag)
    return plaintext.decode()
