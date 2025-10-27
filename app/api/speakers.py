"""
Speakers API router for SecureTranscribe.
Handles speaker management, profile creation, and voice matching endpoints.
"""

import logging
from typing import Optional, List, Dict, Any
from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Body,
)
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field

from app.core.database import get_database
from app.models.speaker import Speaker
from app.models.session import UserSession
from app.services.speaker_service import SpeakerService
from app.services.audio_processor import AudioProcessor
from app.utils.exceptions import (
    SecureTranscribeError,
    ValidationError,
    SpeakerError,
)
from app.api.transcription import get_current_session

logger = logging.getLogger(__name__)
router = APIRouter()


# Pydantic models for request/response
class SpeakerCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    gender: Optional[str] = Field(None, pattern="^(male|female|unknown)$")
    age_range: Optional[str] = Field(None, pattern="^(child|young_adult|adult|senior)$")
    language: str = Field(default="en", max_length=10)
    accent: Optional[str] = Field(None, max_length=100)
    description: Optional[str] = Field(None, max_length=1000)


class SpeakerUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    gender: Optional[str] = Field(None, pattern="^(male|female|unknown)$")
    age_range: Optional[str] = Field(None, pattern="^(child|young_adult|adult|senior)$")
    language: Optional[str] = Field(None, max_length=10)
    accent: Optional[str] = Field(None, max_length=100)
    description: Optional[str] = Field(None, max_length=1000)
    is_verified: Optional[bool] = None


class SpeakerResponse(BaseModel):
    id: int
    name: str
    display_name: str
    gender: Optional[str]
    age_range: Optional[str]
    language: str
    accent: Optional[str]
    confidence_score: float
    confidence_level: str
    is_verified: bool
    is_active: bool
    sample_count: int
    has_voice_data: bool
    created_at: Optional[str]
    updated_at: Optional[str]
    description: Optional[str]

    class Config:
        from_attributes = True


@router.get("/", response_model=List[SpeakerResponse])
async def list_speakers(
    active_only: bool = True,
    verified_only: bool = False,
    page: int = 1,
    per_page: int = 50,
    db: Session = Depends(get_database),
    user_session: UserSession = Depends(get_current_session),
):
    """
    List speakers with optional filtering.

    Args:
        active_only: Only return active speakers
        verified_only: Only return verified speakers
        page: Page number
        per_page: Results per page
        db: Database session
        user_session: Current user session

    Returns:
        List of speakers
    """
    try:
        speaker_service = SpeakerService()
        speakers = speaker_service.get_all_speakers(
            db, active_only, verified_only, page, per_page
        )

        return [SpeakerResponse.from_orm(speaker) for speaker in speakers]

    except SecureTranscribeError:
        raise
    except Exception as e:
        logger.error(f"Failed to list speakers: {e}")
        raise SpeakerError(f"Failed to list speakers: {str(e)}")


@router.get("/{speaker_id}", response_model=SpeakerResponse)
async def get_speaker(
    speaker_id: int,
    db: Session = Depends(get_database),
    user_session: UserSession = Depends(get_current_session),
):
    """
    Get detailed information about a specific speaker.

    Args:
        speaker_id: ID of the speaker
        db: Database session
        user_session: Current user session

    Returns:
        Speaker information
    """
    try:
        speaker = db.query(Speaker).filter(Speaker.id == speaker_id).first()

        if not speaker:
            raise HTTPException(status_code=404, detail="Speaker not found")

        return SpeakerResponse.from_orm(speaker)

    except SecureTranscribeError:
        raise
    except Exception as e:
        logger.error(f"Failed to get speaker: {e}")
        raise SpeakerError(f"Failed to get speaker: {str(e)}")


@router.post("/", response_model=SpeakerResponse)
async def create_speaker(
    speaker_data: SpeakerCreate,
    db: Session = Depends(get_database),
    user_session: UserSession = Depends(get_current_session),
):
    """
    Create a new speaker profile.

    Args:
        speaker_data: Speaker creation data
        db: Database session
        user_session: Current user session

    Returns:
        Created speaker information
    """
    try:
        speaker_service = SpeakerService()
        speaker = speaker_service.create_speaker(
            db=db,
            name=speaker_data.name,
            gender=speaker_data.gender,
            age_range=speaker_data.age_range,
            language=speaker_data.language,
            accent=speaker_data.accent,
            description=speaker_data.description,
        )

        return SpeakerResponse.from_orm(speaker)

    except SecureTranscribeError:
        raise
    except Exception as e:
        logger.error(f"Failed to create speaker: {e}")
        raise SpeakerError(f"Failed to create speaker: {str(e)}")


@router.put("/{speaker_id}", response_model=SpeakerResponse)
async def update_speaker(
    speaker_id: int,
    speaker_data: SpeakerUpdate,
    db: Session = Depends(get_database),
    user_session: UserSession = Depends(get_current_session),
):
    """
    Update speaker profile information.

    Args:
        speaker_id: ID of the speaker to update
        speaker_data: Updated speaker data
        db: Database session
        user_session: Current user session

    Returns:
        Updated speaker information
    """
    try:
        speaker_service = SpeakerService()
        speaker = speaker_service.update_speaker(
            db=db,
            speaker_id=speaker_id,
            name=speaker_data.name,
            gender=speaker_data.gender,
            age_range=speaker_data.age_range,
            language=speaker_data.language,
            accent=speaker_data.accent,
            description=speaker_data.description,
            is_verified=speaker_data.is_verified,
        )

        return SpeakerResponse.from_orm(speaker)

    except SecureTranscribeError:
        raise
    except Exception as e:
        logger.error(f"Failed to update speaker: {e}")
        raise SpeakerError(f"Failed to update speaker: {str(e)}")


@router.delete("/{speaker_id}")
async def delete_speaker(
    speaker_id: int,
    permanent: bool = False,
    db: Session = Depends(get_database),
    user_session: UserSession = Depends(get_current_session),
):
    """
    Delete or deactivate a speaker.

    Args:
        speaker_id: ID of the speaker to delete
        permanent: If True, permanently delete; if False, deactivate
        db: Database session
        user_session: Current user session

    Returns:
        Deletion result
    """
    try:
        speaker_service = SpeakerService()
        success = speaker_service.delete_speaker(db, speaker_id, permanent)

        if not success:
            raise HTTPException(status_code=404, detail="Speaker not found")

        return {
            "success": True,
            "message": "Speaker deleted successfully"
            if permanent
            else "Speaker deactivated",
            "speaker_id": speaker_id,
            "permanent": permanent,
        }

    except SecureTranscribeError:
        raise
    except Exception as e:
        logger.error(f"Failed to delete speaker: {e}")
        raise SpeakerError(f"Failed to delete speaker: {str(e)}")


@router.get("/{speaker_id}/statistics")
async def get_speaker_statistics(
    speaker_id: int,
    db: Session = Depends(get_database),
    user_session: UserSession = Depends(get_current_session),
):
    """
    Get detailed statistics for a speaker.

    Args:
        speaker_id: ID of the speaker
        db: Database session
        user_session: Current user session

    Returns:
        Speaker statistics
    """
    try:
        speaker_service = SpeakerService()
        stats = speaker_service.get_speaker_statistics(db, speaker_id)

        return stats

    except SecureTranscribeError:
        raise
    except Exception as e:
        logger.error(f"Failed to get speaker statistics: {e}")
        raise SpeakerError(f"Failed to get speaker statistics: {str(e)}")


@router.post("/search")
async def search_speakers(
    query: str = Body(..., embed=True),
    active_only: bool = True,
    db: Session = Depends(get_database),
    user_session: UserSession = Depends(get_current_session),
):
    """
    Search speakers by name or description.

    Args:
        query: Search query
        active_only: Only return active speakers
        db: Database session
        user_session: Current user session

    Returns:
        List of matching speakers
    """
    try:
        if not query or len(query.strip()) < 2:
            raise ValidationError("Search query must be at least 2 characters long")

        speaker_service = SpeakerService()
        speakers = speaker_service.search_speakers(db, query.strip(), active_only)

        return [SpeakerResponse.from_orm(speaker) for speaker in speakers]

    except SecureTranscribeError:
        raise
    except Exception as e:
        logger.error(f"Failed to search speakers: {e}")
        raise SpeakerError(f"Failed to search speakers: {str(e)}")


@router.post("/match")
async def find_matching_speakers(
    audio_file_path: str = Body(..., embed=True),
    min_similarity: float = Body(0.7, embed=True),
    max_results: int = Body(5, embed=True),
    db: Session = Depends(get_database),
    user_session: UserSession = Depends(get_current_session),
):
    """
    Find speakers that match a provided audio file.

    Args:
        audio_file_path: Path to audio file for matching
        min_similarity: Minimum similarity threshold (0.0-1.0)
        max_results: Maximum number of results to return
        db: Database session
        user_session: Current user session

    Returns:
        List of matching speakers with similarity scores
    """
    try:
        if not (0.0 <= min_similarity <= 1.0):
            raise ValidationError("min_similarity must be between 0.0 and 1.0")

        if max_results < 1 or max_results > 50:
            raise ValidationError("max_results must be between 1 and 50")

        speaker_service = SpeakerService()
        matches = speaker_service.find_matching_speakers(
            db, audio_file_path, min_similarity, max_results
        )

        result = []
        for speaker, similarity_score in matches:
            speaker_data = SpeakerResponse.from_orm(speaker)
            speaker_data.similarity_score = similarity_score
            result.append(speaker_data.dict())

        return {
            "matches": result,
            "total_matches": len(result),
            "min_similarity_used": min_similarity,
            "audio_file": audio_file_path,
        }

    except SecureTranscribeError:
        raise
    except Exception as e:
        logger.error(f"Failed to find matching speakers: {e}")
        raise SpeakerError(f"Failed to find matching speakers: {str(e)}")


@router.post("/{speaker_id}/add-voice-sample")
async def add_voice_sample(
    speaker_id: int,
    audio_file_path: str = Body(..., embed=True),
    db: Session = Depends(get_database),
    user_session: UserSession = Depends(get_current_session),
):
    """
    Add a voice sample to improve speaker profile.

    Args:
        speaker_id: ID of the speaker
        audio_file_path: Path to audio file
        db: Database session
        user_session: Current user session

    Returns:
        Updated speaker information
    """
    try:
        speaker_service = SpeakerService()
        speaker = speaker_service.add_voice_sample(db, speaker_id, audio_file_path)

        return SpeakerResponse.from_orm(speaker)

    except SecureTranscribeError:
        raise
    except Exception as e:
        logger.error(f"Failed to add voice sample: {e}")
        raise SpeakerError(f"Failed to add voice sample: {str(e)}")


@router.post("/merge")
async def merge_speakers(
    source_speaker_id: int = Body(..., embed=True),
    target_speaker_id: int = Body(..., embed=True),
    db: Session = Depends(get_database),
    user_session: UserSession = Depends(get_current_session),
):
    """
    Merge two speaker profiles, keeping the target as primary.

    Args:
        source_speaker_id: ID of speaker to merge from
        target_speaker_id: ID of speaker to merge into
        db: Database session
        user_session: Current user session

    Returns:
        Merged speaker information
    """
    try:
        if source_speaker_id == target_speaker_id:
            raise ValidationError("Source and target speaker IDs must be different")

        speaker_service = SpeakerService()
        merged_speaker = speaker_service.merge_speakers(
            db, source_speaker_id, target_speaker_id
        )

        return SpeakerResponse.from_orm(merged_speaker)

    except SecureTranscribeError:
        raise
    except Exception as e:
        logger.error(f"Failed to merge speakers: {e}")
        raise SpeakerError(f"Failed to merge speakers: {str(e)}")
