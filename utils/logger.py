"""
Logging utility for GeoAI-GVI project
Provides centralized logging configuration and management
"""

import logging
import os
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colored output for console"""
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m'        # Reset
    }
    
    def format(self, record):
        # Add color to levelname
        if record.levelname in self.COLORS:
            record.levelname = (
                f"{self.COLORS[record.levelname]}"
                f"{record.levelname}"
                f"{self.COLORS['RESET']}"
            )
        
        return super().format(record)


def setup_logger(
    name: str,
    log_level: str = "INFO",
    log_dir: Optional[str] = None,
    console_output: bool = True,
    file_output: bool = True,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Setup logger with file and console handlers
    
    Args:
        name: Logger name (usually module name)
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory for log files (default: ./logs)
        console_output: Enable console output
        file_output: Enable file output
        format_string: Custom format string
        
    Returns:
        Configured logger instance
    """
    
    # Create logger
    logger = logging.getLogger(name)
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    # Set level
    log_level = getattr(logging, log_level.upper(), logging.INFO)
    logger.setLevel(log_level)
    
    # Default format
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        
        # Use colored formatter for console
        console_formatter = ColoredFormatter(format_string)
        console_handler.setFormatter(console_formatter)
        
        logger.addHandler(console_handler)
    
    # File handler
    if file_output:
        # Create log directory
        if log_dir is None:
            log_dir = "./logs"
        
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        
        # Create log file name with timestamp
        timestamp = datetime.now().strftime("%Y%m%d")
        log_file = log_path / f"{name}_{timestamp}.log"
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(log_level)
        
        # Use standard formatter for file (no colors)
        file_formatter = logging.Formatter(format_string)
        file_handler.setFormatter(file_formatter)
        
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get existing logger or create new one with default settings
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        # Setup with default settings if not already configured
        logger = setup_logger(name)
    
    return logger


class LoggerManager:
    """
    Centralized logger management for the project
    """
    
    def __init__(self, log_dir: str = "./logs", default_level: str = "INFO"):
        """
        Initialize logger manager
        
        Args:
            log_dir: Directory for log files
            default_level: Default logging level
        """
        self.log_dir = Path(log_dir)
        self.default_level = default_level
        self._loggers = {}
        
        # Create log directory
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup main project logger
        self.main_logger = self.get_logger("geoai_gvi")
    
    def get_logger(self, name: str, level: Optional[str] = None) -> logging.Logger:
        """
        Get or create logger with consistent configuration
        
        Args:
            name: Logger name
            level: Logging level (uses default if None)
            
        Returns:
            Configured logger
        """
        if name in self._loggers:
            return self._loggers[name]
        
        # Create new logger
        logger = setup_logger(
            name=name,
            log_level=level or self.default_level,
            log_dir=str(self.log_dir),
            console_output=True,
            file_output=True
        )
        
        self._loggers[name] = logger
        return logger
    
    def set_level(self, level: str):
        """
        Set logging level for all managed loggers
        
        Args:
            level: New logging level
        """
        log_level = getattr(logging, level.upper(), logging.INFO)
        
        for logger in self._loggers.values():
            logger.setLevel(log_level)
            
            # Update handler levels
            for handler in logger.handlers:
                handler.setLevel(log_level)
    
    def log_system_info(self):
        """Log system information for debugging"""
        import platform
        import psutil
        
        logger = self.main_logger
        
        logger.info("=== System Information ===")
        logger.info(f"Platform: {platform.platform()}")
        logger.info(f"Python: {platform.python_version()}")
        logger.info(f"CPU Count: {psutil.cpu_count()}")
        logger.info(f"Memory: {psutil.virtual_memory().total / (1024**3):.1f} GB")
        logger.info("=" * 30)
    
    def log_config_info(self, config_dict: dict):
        """
        Log configuration information
        
        Args:
            config_dict: Configuration dictionary to log
        """
        logger = self.main_logger
        
        logger.info("=== Configuration ===")
        for key, value in config_dict.items():
            if 'key' in key.lower() or 'password' in key.lower():
                # Mask sensitive information
                logger.info(f"{key}: ***masked***")
            else:
                logger.info(f"{key}: {value}")
        logger.info("=" * 30)
    
    def cleanup_old_logs(self, days: int = 30):
        """
        Clean up log files older than specified days
        
        Args:
            days: Number of days to keep logs
        """
        import time
        
        cutoff_time = time.time() - (days * 24 * 60 * 60)
        removed_count = 0
        
        for log_file in self.log_dir.glob("*.log"):
            if log_file.stat().st_mtime < cutoff_time:
                try:
                    log_file.unlink()
                    removed_count += 1
                except Exception as e:
                    self.main_logger.warning(f"Could not remove old log file {log_file}: {e}")
        
        if removed_count > 0:
            self.main_logger.info(f"Cleaned up {removed_count} old log files")


# Global logger manager instance
_logger_manager = None


def get_project_logger(name: str) -> logging.Logger:
    """
    Get project logger with consistent configuration
    
    Args:
        name: Logger name (usually __name__)
        
    Returns:
        Configured logger
    """
    global _logger_manager
    
    if _logger_manager is None:
        # Initialize with environment variables or defaults
        log_dir = os.environ.get('LOG_DIR', './logs')
        log_level = os.environ.get('LOG_LEVEL', 'INFO')
        _logger_manager = LoggerManager(log_dir, log_level)
    
    return _logger_manager.get_logger(name)


def set_project_log_level(level: str):
    """
    Set logging level for entire project
    
    Args:
        level: Logging level
    """
    global _logger_manager
    
    if _logger_manager is not None:
        _logger_manager.set_level(level)


# Convenience decorators
def log_execution_time(logger: Optional[logging.Logger] = None):
    """
    Decorator to log function execution time
    
    Args:
        logger: Logger to use (creates default if None)
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            import time
            
            if logger is None:
                log = get_project_logger(func.__module__)
            else:
                log = logger
            
            start_time = time.time()
            log.debug(f"Starting {func.__name__}")
            
            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                log.debug(f"Completed {func.__name__} in {execution_time:.2f}s")
                return result
            except Exception as e:
                execution_time = time.time() - start_time
                log.error(f"Failed {func.__name__} after {execution_time:.2f}s: {e}")
                raise
        
        return wrapper
    return decorator


def log_errors(logger: Optional[logging.Logger] = None):
    """
    Decorator to log function errors
    
    Args:
        logger: Logger to use (creates default if None)
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            if logger is None:
                log = get_project_logger(func.__module__)
            else:
                log = logger
            
            try:
                return func(*args, **kwargs)
            except Exception as e:
                log.error(f"Error in {func.__name__}: {e}", exc_info=True)
                raise
        
        return wrapper
    return decorator


# Test function
def test_logger():
    """Test logging functionality"""
    
    # Test basic logger setup
    logger = setup_logger("test_logger", log_level="DEBUG")
    
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    
    # Test project logger
    project_logger = get_project_logger("test_module")
    project_logger.info("Project logger test")
    
    # Test decorated function
    @log_execution_time()
    @log_errors()
    def test_function():
        import time
        time.sleep(0.1)
        return "success"
    
    result = test_function()
    
    print("✓ Logger test completed successfully")
    return True


# if __name__ == "__main__":
#    test_logger()
