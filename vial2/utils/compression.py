import zlib
from fastapi import HTTPException
from ..error_logging.error_log import error_logger
import logging

logger = logging.getLogger(__name__)

class DataCompressor:
    async def compress_data(self, data: bytes):
        try:
            compressed = zlib.compress(data)
            return {"status": "success", "compressed_data": compressed.hex()}
        except Exception as e:
            error_logger.log_error("compression", f"Data compression failed: {str(e)}", str(e.__traceback__))
            logger.error(f"Data compression failed: {str(e)}")
            raise HTTPException(status_code=400, detail=str(e))

    async def decompress_data(self, compressed_data: str):
        try:
            compressed_bytes = bytes.fromhex(compressed_data)
            decompressed = zlib.decompress(compressed_bytes)
            return {"status": "success", "decompressed_data": decompressed.decode()}
        except Exception as e:
            error_logger.log_error("compression", f"Data decompression failed: {str(e)}", str(e.__traceback__))
            logger.error(f"Data decompression failed: {str(e)}")
            raise HTTPException(status_code=400, detail=str(e))

data_compressor = DataCompressor()

# xAI Artifact Tags: #vial2 #utils #compression #neon_mcp
