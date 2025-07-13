import os
import sys
import time
import logging
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path
import threading
from functools import wraps

# Modern OpenAI client import (handle not installed case)
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    OpenAI = None  # So type checker is happy

# Configuration class for better management
@dataclass
class Config:
    """Centralized configuration management."""
    openai_api_key: str = field(default_factory=lambda: os.environ.get("OPENAI_API_KEY", ""))
    openai_model: str = field(default_factory=lambda: os.environ.get("OPENAI_MODEL", "gpt-4-1106-preview"))
    log_level: str = field(default_factory=lambda: os.environ.get("LOG_LEVEL", "INFO"))
    max_tokens_default: int = 32768
    temperature_default: float = 0.2
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    max_content_length: int = 400000
    max_prompt_length: int = 400000 
    rate_limit_rpm: int = 60  # requests per minute
    corpus_path: str = field(default_factory=lambda: os.path.join(os.path.dirname(os.path.abspath(__file__)), "corpus"))
    manual_outfile: str = "FINAL_USER_MANUAL.md"  # Output filename for the manual

    # Coverage/QA flags for downstream use
    enable_coverage_tracking: bool = True
    enable_ai_qa_check: bool = True

    def __post_init__(self):
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        if self.log_level not in ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']:
            self.log_level = 'INFO'
        Path(self.corpus_path).mkdir(parents=True, exist_ok=True)

# Global configuration instance
config = Config()

def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    logger = logging.getLogger("knowledge_graph")
    logger.setLevel(getattr(logging, log_level.upper()))
    logger.handlers.clear()
    formatter = logging.Formatter(
        fmt="[%(asctime)s] %(name)s.%(funcName)s:%(lineno)d %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger

# Global logger
logger = setup_logging(config.log_level)

class RateLimiter:
    """Thread-safe rate limiter for API calls."""
    def __init__(self, max_calls: int, time_window: int = 60):
        self.max_calls = max_calls
        self.time_window = time_window
        self.calls = []
        self.lock = threading.Lock()
    def __call__(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with self.lock:
                now = time.time()
                self.calls = [call_time for call_time in self.calls if now - call_time < self.time_window]
                if len(self.calls) >= self.max_calls:
                    sleep_time = self.time_window - (now - self.calls[0])
                    if sleep_time > 0:
                        logger.warning(f"Rate limit reached. Sleeping for {sleep_time:.2f} seconds")
                        time.sleep(sleep_time)
                self.calls.append(now)
            return func(*args, **kwargs)
        return wrapper

def initialize_openai_client() -> Optional[OpenAI]:
    if not OPENAI_AVAILABLE:
        logger.error("OpenAI library not available. Please install: pip install openai")
        return None
    if not config.openai_api_key or config.openai_api_key == "YOUR_API_KEY_HERE":
        logger.error("Valid OpenAI API key required. Set OPENAI_API_KEY environment variable.")
        return None
    try:
        client = OpenAI(api_key=config.openai_api_key)
        client.models.list()
        logger.info(f"OpenAI client initialized successfully with model: {config.openai_model}")
        return client
    except Exception as e:
        logger.error(f"Failed to initialize OpenAI client: {e}")
        return None

openai_client = initialize_openai_client()

def validate_environment():
    issues = []
    if not OPENAI_AVAILABLE:
        issues.append("OpenAI library not installed")
    if not config.openai_api_key:
        issues.append("OpenAI API key not configured")
    if not Path(config.corpus_path).exists():
        issues.append(f"Corpus directory not found: {config.corpus_path}")
    try:
        test_file = Path(config.corpus_path) / "test_write.tmp"
        test_file.write_text("test")
        test_file.unlink()
    except Exception:
        issues.append("No write permissions in corpus directory")
    if issues:
        logger.error("Environment validation failed:")
        for issue in issues:
            logger.error(f"  - {issue}")
        return False
    logger.info("Environment validation passed")
    return True

# Optionally, add auto-validation for import (optional, not required)
if __name__ != "__main__":
    logger.info(f"Knowledge Graph System initialized - Model: {config.openai_model}, Log Level: {config.log_level}")
    if not validate_environment():
        logger.error("Environment validation failed. Some features may not work correctly.")
    else:
        logger.info("System ready for operation")
