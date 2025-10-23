"""
Sessions API router for SecureTranscribe.
Handles session management, authentication, and user tracking endpoints.
"""

import logging
from typing import Optional, Dict, Any
from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Body,
)
from sqlalchemy.orm import Session
from pydantic import BaseModel

from ..core.database import get_database
from ..models.session import UserSession
from ..utils.exceptions import (
    SecureTranscribeError,
    ValidationError,
    AuthenticationError,
)

logger = logging.getLogger(__name__)
router = APIRouter()


# Pydantic models for request/response
class SessionCreate(BaseModel):
    user_identifier: Optional[str] = None
    user_agent: Optional[str] = None
    ip_address: Optional[str] = None


class SessionUpdate(BaseModel):
    user_identifier: Optional[str] = None
    preferences: Optional[Dict[str, Any]] = None


class SessionResponse(BaseModel):
    id: int
    session_id: str
    user_identifier: Optional[str]
    created_at: Optional[str]
    last_accessed: Optional[str]
    expires_at: Optional[str]
    is_active: bool
    is_authenticated: bool
    is_valid: bool
    queue_position: int
    is_processing: bool
    total_files_processed: int
    session_age: float
    formatted_session_age: str
    processing_efficiency: float
    average_confidence: float

    class Config:
        from_attributes = True


@router.get("/current", response_model=SessionResponse)
async def get_current_session_info(
    request,
    db: Session = Depends(get_database),
):
    """
    Get information about the current user session.

    Args:
        request: FastAPI request object
        db: Database session

    Returns:
        Current session information
    """
    try:
        session_token = request.session.get("session_token")

        if not session_token:
            raise AuthenticationError("No active session found")

        user_session = UserSession.get_by_token(db, session_token)

        if not user_session or not user_session.is_valid:
            raise AuthenticationError("Invalid or expired session")

        user_session.update_last_accessed()
        db.commit()

        return SessionResponse.from_orm(user_session)

    except SecureTranscribeError:
        raise
    except Exception as e:
        logger.error(f"Failed to get current session: {e}")
        raise AuthenticationError(f"Failed to get session: {str(e)}")


@router.post("/create", response_model=SessionResponse)
async def create_session(
    session_data: SessionCreate,
    db: Session = Depends(get_database),
):
    """
    Create a new user session.

    Args:
        session_data: Session creation data
        db: Database session

    Returns:
        Created session information
    """
    try:
        user_session = UserSession.create_session(
            db=db,
            user_identifier=session_data.user_identifier,
            user_agent=session_data.user_agent,
            ip_address=session_data.ip_address,
        )

        return SessionResponse.from_orm(user_session)

    except SecureTranscribeError:
        raise
    except Exception as e:
        logger.error(f"Failed to create session: {e}")
        raise AuthenticationError(f"Failed to create session: {str(e)}")


@router.put("/current", response_model=SessionResponse)
async def update_current_session(
    request,
    session_data: SessionUpdate,
    db: Session = Depends(get_database),
):
    """
    Update the current user session.

    Args:
        request: FastAPI request object
        session_data: Session update data
        db: Database session

    Returns:
        Updated session information
    """
    try:
        session_token = request.session.get("session_token")

        if not session_token:
            raise AuthenticationError("No active session found")

        user_session = UserSession.get_by_token(db, session_token)

        if not user_session or not user_session.is_valid:
            raise AuthenticationError("Invalid or expired session")

        # Update session data
        if session_data.user_identifier is not None:
            user_session.user_identifier = session_data.user_identifier

        if session_data.preferences is not None:
            user_session.preferences = session_data.preferences

        user_session.update_last_accessed()
        db.commit()

        return SessionResponse.from_orm(user_session)

    except SecureTranscribeError:
        raise
    except Exception as e:
        logger.error(f"Failed to update session: {e}")
        raise AuthenticationError(f"Failed to update session: {str(e)}")


@router.delete("/current")
async def invalidate_current_session(
    request,
    db: Session = Depends(get_database),
):
    """
    Invalidate the current user session.

    Args:
        request: FastAPI request object
        db: Database session

    Returns:
        Invalidation result
    """
    try:
        session_token = request.session.get("session_token")

        if not session_token:
            raise AuthenticationError("No active session found")

        user_session = UserSession.get_by_token(db, session_token)

        if user_session:
            user_session.invalidate()
            db.commit()

        # Clear session cookie
        request.session.clear()

        return {"success": True, "message": "Session invalidated successfully"}

    except SecureTranscribeError:
        raise
    except Exception as e:
        logger.error(f"Failed to invalidate session: {e}")
        raise AuthenticationError(f"Failed to invalidate session: {str(e)}")


@router.post("/extend")
async def extend_current_session(
    request,
    hours: int = Body(1, embed=True),
    db: Session = Depends(get_database),
):
    """
    Extend the current session expiry.

    Args:
        request: FastAPI request object
        hours: Number of hours to extend session
        db: Database session

    Returns:
        Extended session information
    """
    try:
        if hours < 1 or hours > 24:
            raise ValidationError("Hours must be between 1 and 24")

        session_token = request.session.get("session_token")

        if not session_token:
            raise AuthenticationError("No active session found")

        user_session = UserSession.get_by_token(db, session_token)

        if not user_session or not user_session.is_valid:
            raise AuthenticationError("Invalid or expired session")

        user_session.extend_session(hours)
        db.commit()

        return {
            "success": True,
            "message": f"Session extended by {hours} hours",
            "expires_at": user_session.expires_at.isoformat()
            if user_session.expires_at
            else None,
            "time_until_expiry": user_session.time_until_expiry,
        }

    except SecureTranscribeError:
        raise
    except Exception as e:
        logger.error(f"Failed to extend session: {e}")
        raise AuthenticationError(f"Failed to extend session: {str(e)}")


@router.get("/statistics")
async def get_session_statistics(
    request,
    db: Session = Depends(get_database),
):
    """
    Get global session statistics (admin function).

    Args:
        request: FastAPI request object
        db: Database session

    Returns:
        Session statistics
    """
    try:
        # This could be protected by admin authentication in a real implementation
        stats = UserSession.get_session_statistics(db)

        return stats

    except Exception as e:
        logger.error(f"Failed to get session statistics: {e}")
        raise ValidationError(f"Failed to get session statistics: {str(e)}")


@router.get("/preferences")
async def get_session_preferences(
    request,
    db: Session = Depends(get_database),
):
    """
    Get current session preferences.

    Args:
        request: FastAPI request object
        db: Database session

    Returns:
        Session preferences
    """
    try:
        session_token = request.session.get("session_token")

        if not session_token:
            raise AuthenticationError("No active session found")

        user_session = UserSession.get_by_token(db, session_token)

        if not user_session or not user_session.is_valid:
            raise AuthenticationError("Invalid or expired session")

        return {
            "preferences": user_session.preferences or {},
            "user_identifier": user_session.user_identifier,
        }

    except SecureTranscribeError:
        raise
    except Exception as e:
        logger.error(f"Failed to get session preferences: {e}")
        raise AuthenticationError(f"Failed to get session preferences: {str(e)}")


@router.post("/preferences")
async def update_session_preferences(
    request,
    preferences: Dict[str, Any] = Body(..., embed=True),
    db: Session = Depends(get_database),
):
    """
    Update current session preferences.

    Args:
        request: FastAPI request object
        preferences: Preferences to update
        db: Database session

    Returns:
        Updated preferences
    """
    try:
        session_token = request.session.get("session_token")

        if not session_token:
            raise AuthenticationError("No active session found")

        user_session = UserSession.get_by_token(db, session_token)

        if not user_session or not user_session.is_valid:
            raise AuthenticationError("Invalid or expired session")

        # Update preferences
        if user_session.preferences is None:
            user_session.preferences = {}

        user_session.preferences.update(preferences)
        user_session.update_last_accessed()
        db.commit()

        return {
            "success": True,
            "preferences": user_session.preferences,
            "message": "Preferences updated successfully",
        }

    except SecureTranscribeError:
        raise
    except Exception as e:
        logger.error(f"Failed to update session preferences: {e}")
        raise AuthenticationError(f"Failed to update session preferences: {str(e)}")
