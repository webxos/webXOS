import pytest
from ..utils.compression import data_compressor
from ..error_logging.error_log import error_logger

@pytest.mark.asyncio
async def test_compress_decompress():
    try:
        data = "Test data for compression".encode()
        compressed_result = await data_compressor.compress_data(data)
        assert compressed_result["status"] == "success"
        decompressed_result = await data_compressor.decompress_data(compressed_result["compressed_data"])
        assert decompressed_result["status"] == "success"
        assert decompressed_result["decompressed_data"] == data.decode()
    except Exception as e:
        error_logger.log_error("test_compression", f"Test compress/decompress failed: {str(e)}", str(e.__traceback__))
        raise

# xAI Artifact Tags: #vial2 #tests #compression #neon_mcp
