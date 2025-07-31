import os
import time
from apscheduler.schedulers.background import BackgroundScheduler

from inference.server.config import IMAGE_DIR, IMAGE_RETENTION_HOURS


class CleanupService:
    """Service for managing cleanup of old generated images."""

    def __init__(self):
        self.scheduler = BackgroundScheduler()

    def cleanup_old_images(self):
        """Delete images older than the retention period"""
        print(
            f"Running cleanup job: removing images older than {IMAGE_RETENTION_HOURS} hours"
        )
        cutoff_time = time.time() - (IMAGE_RETENTION_HOURS * 3600)
        deleted_count = 0

        for filename in os.listdir(IMAGE_DIR):
            file_path = os.path.join(IMAGE_DIR, filename)
            if os.path.isfile(file_path):
                file_creation_time = os.path.getctime(file_path)
                if file_creation_time < cutoff_time:
                    try:
                        os.remove(file_path)
                        deleted_count += 1
                    except Exception as e:
                        print(f"Error deleting {file_path}: {e}")

        print(f"Cleanup complete: {deleted_count} images removed")

    def start(self):
        """Start the cleanup scheduler."""
        self.scheduler.add_job(
            self.cleanup_old_images, "interval", hours=1, id="cleanup_job"
        )
        self.scheduler.start()
        print("Background cleanup job scheduler started")

    def shutdown(self):
        """Shutdown the cleanup scheduler."""
        self.scheduler.shutdown()
        print("Background cleanup job scheduler shut down")


# Create a singleton instance
cleanup_service = CleanupService()
