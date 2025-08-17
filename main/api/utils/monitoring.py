import psutil
import asyncio
from fastapi import HTTPException
from ..config.redis_config import get_redis
from ..utils.logging import log_error, log_info

class SystemMonitor:
    def __init__(self):
        self.redis = None

    async def connect(self):
        self.redis = await get_redis()

    async def collect_metrics(self):
        try:
            cpu_usage = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            process = psutil.Process()
            metrics = {
                "cpu_usage_percent": cpu_usage,
                "memory_used_mb": memory.used / 1024 / 1024,
                "memory_total_mb": memory.total / 1024 / 1024,
                "disk_usage_percent": disk.percent,
                "process_memory_mb": process.memory_info().rss / 1024 / 1024,
                "timestamp": psutil.boot_time()
            }
            await self.redis.set("system_metrics", json.dumps(metrics), ex=3600)
            log_info(f"Metrics collected: {json.dumps(metrics)}")
            return metrics
        except Exception as e:
            log_error(f"Metrics collection failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Monitoring error: {str(e)}")

    async def monitor_loop(self):
        await self.connect()
        while True:
            await self.collect_metrics()
            await asyncio.sleep(60)

monitor = SystemMonitor()

async def start_monitoring():
    asyncio.create_task(monitor.monitor_loop())
