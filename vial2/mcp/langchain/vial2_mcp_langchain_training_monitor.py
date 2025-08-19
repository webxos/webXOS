import asyncio

class TrainingMonitor:
    async def check_progress(self, epoch):
        await asyncio.sleep(0.1)  # Simulate monitoring
        return {"epoch": epoch, "status": "in_progress"}