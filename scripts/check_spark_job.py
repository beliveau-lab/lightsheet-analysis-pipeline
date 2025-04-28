# scripts/check_spark_job.py
import argparse
import time
import sys
import os
import glob
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('SparkJobCheck')

def find_latest_spark_log_dir(base_log_dir: str) -> str | None:
    """Finds the most recent directory matching '2*' in the base log dir."""
    try:
        # Find directories starting with '2' (like '2024...')
        possible_dirs = [
            p for p in Path(base_log_dir).glob('2*') if p.is_dir()
        ]
        if not possible_dirs:
            logger.warning(f"No log directories starting with '2*' found in {base_log_dir}")
            return None
        # Sort by modification time (or name as fallback)
        latest_dir = max(possible_dirs, key=lambda p: p.stat().st_mtime)
        # Alternative sort by name if mod time is unreliable:
        # latest_dir = max(possible_dirs, key=lambda p: p.name)
        logger.info(f"Found latest log directory: {latest_dir}")
        return str(latest_dir)
    except FileNotFoundError:
        logger.error(f"Base log directory not found: {base_log_dir}")
        return None
    except Exception as e:
        logger.error(f"Error finding latest log directory in {base_log_dir}: {e}", exc_info=True)
        return None

def wait_for_shutdown_log(spark_log_dir: str, timeout: int, interval: int) -> bool:
    """Waits for the 12-shutdown.log file to appear."""
    shutdown_log = Path(spark_log_dir) / "logs" / "12-shutdown.log"
    start_time = time.time()
    while not shutdown_log.exists():
        elapsed = time.time() - start_time
        if elapsed >= timeout:
            logger.error(f"Timeout ({timeout}s) waiting for shutdown log: {shutdown_log}")
            return False
        logger.info(f"Waiting for Spark job completion (shutdown log)... ({int(elapsed)}s elapsed)")
        time.sleep(interval)
    logger.info(f"Shutdown log found: {shutdown_log}")
    return True

def check_driver_log(spark_log_dir: str, success_pattern: str) -> bool:
    """Checks the 04-driver.log for a success pattern."""
    driver_log = Path(spark_log_dir) / "logs" / "04-driver.log"
    if not driver_log.exists():
        logger.error(f"Driver log not found: {driver_log}")
        # Optionally list available logs for debugging
        log_dir_path = Path(spark_log_dir) / "logs"
        if log_dir_path.exists():
             available_logs = [p.name for p in log_dir_path.iterdir()]
             logger.info(f"Available logs in {log_dir_path}: {available_logs}")
        return False

    logger.info(f"Checking driver log: {driver_log} for pattern: '{success_pattern}'")
    try:
        with open(driver_log, 'r') as f:
            # Read line by line or whole file depending on expected size/pattern location
            # For "done, took", it's likely near the end. Reading last N lines might be faster.
            # Simple approach: read whole file (can be large)
            content = f.read()
            if success_pattern in content:
                logger.info(f"Success pattern '{success_pattern}' found in driver log.")
                return True
            else:
                logger.error(f"Success pattern '{success_pattern}' *not* found in driver log.")
                # Log last few lines for context
                lines = content.splitlines()
                last_lines = lines[-20:] # Log last 20 lines
                logger.info("Last lines of driver log:")
                for line in last_lines:
                     logger.info(f"  {line}")
                return False
    except Exception as e:
        logger.error(f"Error reading or searching driver log {driver_log}: {e}", exc_info=True)
        return False

def main():
    parser = argparse.ArgumentParser(description="Check status of a Spark job by monitoring its log files.")
    parser.add_argument("--base-log-dir", required=True, help="Base directory containing timestamped Spark log folders.")
    parser.add_argument("--job-name", default="Spark Job", help="Descriptive name for the job being checked (for logging).")
    parser.add_argument("--timeout", type=int, default=3600, help="Maximum time (seconds) to wait for job completion.")
    parser.add_argument("--interval", type=int, default=30, help="Interval (seconds) between checks for completion.")
    parser.add_argument("--success-pattern", default="saving resulting xml", help="Text pattern to search for in the driver log to confirm success.") # Default might need adjustment

    args = parser.parse_args()

    logger.info(f"--- Starting check for {args.job_name} ---")
    logger.info(f"Base log directory: {args.base_log_dir}")
    logger.info(f"Timeout: {args.timeout}s, Interval: {args.interval}s")
    logger.info(f"Success Pattern: '{args.success_pattern}'")

    # 1. Find the relevant log directory
    # We might need to wait briefly for the directory to be created after job submission
    latest_log_dir = None
    find_start_time = time.time()
    find_timeout = 60 # Wait up to 60s for the log dir to appear initially
    while latest_log_dir is None and (time.time() - find_start_time < find_timeout):
        latest_log_dir = find_latest_spark_log_dir(args.base_log_dir)
        if latest_log_dir is None:
             logger.info(f"Log directory not found yet, waiting...")
             time.sleep(5)

    if latest_log_dir is None:
        logger.error(f"Could not find a suitable log directory in {args.base_log_dir} after {find_timeout}s.")
        sys.exit(1)

    # 2. Wait for the shutdown log
    if not wait_for_shutdown_log(latest_log_dir, args.timeout, args.interval):
        logger.error(f"{args.job_name} failed (timeout waiting for shutdown log).")
        sys.exit(1)

    # 3. Check the driver log for success
    # Add a small delay to ensure the driver log is fully written after shutdown appears
    time.sleep(5)
    if not check_driver_log(latest_log_dir, args.success_pattern):
        logger.error(f"{args.job_name} failed (success pattern not found in driver log).")
        sys.exit(1)

    logger.info(f"--- {args.job_name} completed successfully. ---")
    sys.exit(0)

if __name__ == "__main__":
    main() 