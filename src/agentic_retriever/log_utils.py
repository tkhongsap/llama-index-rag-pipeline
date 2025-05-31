"""
Logging Utilities

Provides JSON-lines logging with rotation and compression for agentic retrieval.
"""

import json
import gzip
import time
import os
from pathlib import Path
from typing import Dict, Any, Optional
from functools import wraps
import logging
from logging.handlers import RotatingFileHandler


# Configuration
LOG_DIR = Path("logs")
LOG_FILE = LOG_DIR / "agentic_run.log"
MAX_LOG_SIZE = 10 * 1024 * 1024  # 10 MB
BACKUP_COUNT = 5


def setup_logging():
    """Setup the logging directory and ensure it exists."""
    LOG_DIR.mkdir(exist_ok=True)


class JSONLRotatingFileHandler(RotatingFileHandler):
    """Custom rotating file handler that compresses rotated logs."""
    
    def doRollover(self):
        """Override to compress the rotated log file."""
        if self.stream:
            self.stream.close()
            self.stream = None
        
        if self.backupCount > 0:
            for i in range(self.backupCount - 1, 0, -1):
                sfn = self.rotation_filename(f"{self.baseFilename}.{i}.gz")
                dfn = self.rotation_filename(f"{self.baseFilename}.{i+1}.gz")
                if os.path.exists(sfn):
                    if os.path.exists(dfn):
                        os.remove(dfn)
                    os.rename(sfn, dfn)
            
            # Compress the current log file
            dfn = self.rotation_filename(f"{self.baseFilename}.1.gz")
            if os.path.exists(dfn):
                os.remove(dfn)
            
            # Compress the log file
            with open(self.baseFilename, 'rb') as f_in:
                with gzip.open(dfn, 'wb') as f_out:
                    f_out.writelines(f_in)
            
            # Remove the original file
            os.remove(self.baseFilename)
        
        if not self.delay:
            self.stream = self._open()


def get_logger() -> logging.Logger:
    """Get or create the agentic retrieval logger."""
    logger = logging.getLogger("agentic_retrieval")
    
    if not logger.handlers:
        setup_logging()
        
        # Create custom rotating handler
        handler = JSONLRotatingFileHandler(
            LOG_FILE,
            maxBytes=MAX_LOG_SIZE,
            backupCount=BACKUP_COUNT
        )
        
        # Set formatter to just output the message (we'll format as JSON)
        formatter = logging.Formatter('%(message)s')
        handler.setFormatter(formatter)
        
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    
    return logger


def log_retrieval_call(
    query: str,
    selected_index: str,
    selected_strategy: str,
    latency_ms: float,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    confidence: Optional[float] = None,
    error: Optional[str] = None
):
    """
    Log a retrieval call to the JSON-L log file.
    
    Args:
        query: The user query
        selected_index: The index that was selected
        selected_strategy: The retrieval strategy that was used
        latency_ms: Total latency in milliseconds
        prompt_tokens: Number of prompt tokens used
        completion_tokens: Number of completion tokens generated
        confidence: Confidence score for the selection
        error: Error message if any
    """
    logger = get_logger()
    
    log_entry = {
        "timestamp": time.time(),
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "query": query,
        "index": selected_index,
        "strategy": selected_strategy,
        "latency_ms": round(latency_ms, 2),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens
    }
    
    if confidence is not None:
        log_entry["confidence"] = round(confidence, 3)
    
    if error:
        log_entry["error"] = error
    
    # Log as JSON
    logger.info(json.dumps(log_entry))


def log_retrieval_decorator(func):
    """
    Decorator to automatically log retrieval calls.
    
    The decorated function should return a dict with:
    - response: The actual response
    - metadata: Dict containing index, strategy, latency_ms, etc.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        error = None
        
        try:
            result = func(*args, **kwargs)
            end_time = time.time()
            latency_ms = (end_time - start_time) * 1000
            
            # Extract metadata for logging
            metadata = result.get('metadata', {})
            
            log_retrieval_call(
                query=metadata.get('query', 'unknown'),
                selected_index=metadata.get('index', 'unknown'),
                selected_strategy=metadata.get('strategy', 'unknown'),
                latency_ms=latency_ms,
                prompt_tokens=metadata.get('prompt_tokens', 0),
                completion_tokens=metadata.get('completion_tokens', 0),
                confidence=metadata.get('confidence'),
                error=None
            )
            
            return result
            
        except Exception as e:
            end_time = time.time()
            latency_ms = (end_time - start_time) * 1000
            error = str(e)
            
            # Try to extract query from args/kwargs
            query = "unknown"
            if args and isinstance(args[0], str):
                query = args[0]
            elif 'query' in kwargs:
                query = kwargs['query']
            
            log_retrieval_call(
                query=query,
                selected_index="error",
                selected_strategy="error",
                latency_ms=latency_ms,
                error=error
            )
            
            raise
    
    return wrapper


def read_log_entries(
    log_file: Optional[Path] = None,
    limit: Optional[int] = None
) -> list:
    """
    Read log entries from the JSON-L log file.
    
    Args:
        log_file: Path to log file (defaults to main log)
        limit: Maximum number of entries to read (most recent first)
        
    Returns:
        List of log entry dictionaries
    """
    if log_file is None:
        log_file = LOG_FILE
    
    if not log_file.exists():
        return []
    
    entries = []
    
    try:
        with open(log_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entry = json.loads(line)
                        entries.append(entry)
                    except json.JSONDecodeError:
                        continue
        
        # Sort by timestamp (most recent first)
        entries.sort(key=lambda x: x.get('timestamp', 0), reverse=True)
        
        if limit:
            entries = entries[:limit]
        
        return entries
        
    except Exception as e:
        print(f"Error reading log file: {e}")
        return []


def get_compressed_logs() -> list:
    """Get list of compressed log files."""
    if not LOG_DIR.exists():
        return []
    
    compressed_logs = []
    for file_path in LOG_DIR.glob("*.gz"):
        compressed_logs.append(file_path)
    
    return sorted(compressed_logs)


def read_compressed_log(log_file: Path) -> list:
    """
    Read entries from a compressed log file.
    
    Args:
        log_file: Path to compressed log file
        
    Returns:
        List of log entry dictionaries
    """
    entries = []
    
    try:
        with gzip.open(log_file, 'rt') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entry = json.loads(line)
                        entries.append(entry)
                    except json.JSONDecodeError:
                        continue
        
        return entries
        
    except Exception as e:
        print(f"Error reading compressed log file {log_file}: {e}")
        return []


def cleanup_old_logs(days_to_keep: int = 30):
    """
    Clean up log files older than specified days.
    
    Args:
        days_to_keep: Number of days to keep logs
    """
    if not LOG_DIR.exists():
        return
    
    cutoff_time = time.time() - (days_to_keep * 24 * 60 * 60)
    
    for log_file in LOG_DIR.glob("*.gz"):
        if log_file.stat().st_mtime < cutoff_time:
            try:
                log_file.unlink()
                print(f"Deleted old log file: {log_file}")
            except Exception as e:
                print(f"Error deleting {log_file}: {e}")


# Initialize logging on import
setup_logging() 