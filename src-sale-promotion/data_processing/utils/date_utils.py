"""
Date utilities for Buddhist calendar conversion and date handling.

Buddhist Era (B.E.) to Common Era (C.E.) conversion:
- B.E. year - 543 = C.E. year
- Example: 2568 - 543 = 2025
"""

from datetime import datetime, timedelta
from typing import Optional
import logging
import pandas as pd

logger = logging.getLogger(__name__)


def convert_buddhist_to_gregorian(date_str: str) -> Optional[datetime]:
    """
    Convert Buddhist Era date string to Gregorian datetime object.
    
    Args:
        date_str: Date string in Buddhist calendar (e.g., "2568-10-01 00:00:00")
    
    Returns:
        datetime object in Gregorian calendar, or None if conversion fails
    
    Examples:
        >>> convert_buddhist_to_gregorian("2568-10-01 00:00:00")
        datetime.datetime(2025, 10, 1, 0, 0)
        
        >>> convert_buddhist_to_gregorian("2568-12-31")
        datetime.datetime(2025, 12, 31, 0, 0)
    """
    if not date_str or pd.isna(date_str):
        return None
    
    try:
        # Parse the Buddhist date string
        date_str = str(date_str).strip()
        
        # Extract components manually to avoid pandas datetime overflow
        # Format: "2568-10-01 00:00:00" or "2568-10-01"
        import re
        pattern = r'(\d{4})-(\d{1,2})-(\d{1,2})(?:\s+(\d{1,2}):(\d{1,2}):(\d{1,2}))?'
        match = re.match(pattern, date_str)
        
        if not match:
            logger.warning(f"Date string does not match expected format: '{date_str}'")
            return None
        
        # Extract components
        buddhist_year = int(match.group(1))
        month = int(match.group(2))
        day = int(match.group(3))
        hour = int(match.group(4)) if match.group(4) else 0
        minute = int(match.group(5)) if match.group(5) else 0
        second = int(match.group(6)) if match.group(6) else 0
        
        # Convert Buddhist year to Gregorian
        gregorian_year = buddhist_year - 543
        
        # Create datetime object
        gregorian_date = datetime(gregorian_year, month, day, hour, minute, second)
        
        return gregorian_date
        
    except Exception as e:
        logger.warning(f"Failed to convert Buddhist date '{date_str}': {e}")
        return None


def is_promotion_active(
    start_date: datetime,
    end_date: datetime,
    current_date: Optional[datetime] = None
) -> bool:
    """
    Check if a promotion is currently active.
    
    Args:
        start_date: Promotion start date
        end_date: Promotion end date
        current_date: Date to check against (default: today)
    
    Returns:
        True if promotion is active, False otherwise
    
    Examples:
        >>> start = datetime(2025, 1, 1)
        >>> end = datetime(2025, 12, 31)
        >>> is_promotion_active(start, end, datetime(2025, 6, 15))
        True
    """
    if not start_date or not end_date:
        return False
    
    if current_date is None:
        current_date = datetime.now()
    
    return start_date <= current_date <= end_date


def calculate_duration_days(start_date: datetime, end_date: datetime) -> int:
    """
    Calculate the duration of a promotion in days.
    
    Args:
        start_date: Promotion start date
        end_date: Promotion end date
    
    Returns:
        Number of days in the promotion period
    
    Examples:
        >>> start = datetime(2025, 1, 1)
        >>> end = datetime(2025, 1, 31)
        >>> calculate_duration_days(start, end)
        31
    """
    if not start_date or not end_date:
        return 0
    
    if end_date < start_date:
        logger.warning(f"End date {end_date} is before start date {start_date}")
        return 0
    
    duration = end_date - start_date
    return duration.days + 1  # Include both start and end date


def check_expires_soon(
    end_date: datetime,
    current_date: Optional[datetime] = None,
    threshold_days: int = 7
) -> bool:
    """
    Check if a promotion expires soon (within threshold days).
    
    Args:
        end_date: Promotion end date
        current_date: Date to check from (default: today)
        threshold_days: Number of days to consider as "soon" (default: 7)
    
    Returns:
        True if promotion expires within threshold days, False otherwise
    
    Examples:
        >>> end = datetime(2025, 6, 20)
        >>> current = datetime(2025, 6, 15)
        >>> check_expires_soon(end, current, threshold_days=7)
        True
    """
    if not end_date:
        return False
    
    if current_date is None:
        current_date = datetime.now()
    
    # If already expired, return False
    if end_date < current_date:
        return False
    
    days_until_expiry = (end_date - current_date).days
    return days_until_expiry <= threshold_days


def format_date_for_display(date_obj: Optional[datetime]) -> str:
    """
    Format datetime object for display.
    
    Args:
        date_obj: datetime object to format
    
    Returns:
        Formatted date string (YYYY-MM-DD) or "N/A"
    """
    if not date_obj:
        return "N/A"
    
    return date_obj.strftime('%Y-%m-%d')


def validate_date_range(start_date: datetime, end_date: datetime) -> bool:
    """
    Validate that date range is logical (start before end).
    
    Args:
        start_date: Start date
        end_date: End date
    
    Returns:
        True if valid, False otherwise
    """
    if not start_date or not end_date:
        return False
    
    return start_date <= end_date

