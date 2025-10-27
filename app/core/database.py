"""
Database setup and connection management for SecureTranscribe.
Handles SQLite database initialization, connection pooling, and session management.
"""

import os
import logging
from sqlalchemy import create_engine, MetaData
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool
from typing import Generator

from .config import get_settings, DATABASE_SETTINGS

logger = logging.getLogger(__name__)

# Global variables for database components
engine = None
SessionLocal = None
Base = declarative_base()


def create_database_engine() -> None:
    """Create database engine with appropriate configuration."""
    global engine, SessionLocal

    settings = get_settings()
    database_url = settings.database_url

    # SQLite-specific configuration
    if database_url.startswith("sqlite"):
        engine = create_engine(
            database_url,
            connect_args={
                "check_same_thread": False,
                "timeout": 20,
            },
            poolclass=StaticPool,
            echo=DATABASE_SETTINGS["echo"],
            pool_pre_ping=DATABASE_SETTINGS["pool_pre_ping"],
            pool_recycle=DATABASE_SETTINGS["pool_recycle"],
        )
    else:
        # PostgreSQL/MySQL configuration
        engine = create_engine(
            database_url,
            echo=DATABASE_SETTINGS["echo"],
            pool_pre_ping=DATABASE_SETTINGS["pool_pre_ping"],
            pool_recycle=DATABASE_SETTINGS["pool_recycle"],
        )

    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    logger.info(f"Database engine created for: {database_url}")


def init_database() -> None:
    """Initialize database tables and create required directories."""
    if engine is None:
        create_database_engine()

    # Import all models to ensure they're registered with Base
    from app.models.speaker import Speaker
    from app.models.transcription import Transcription
    from app.models.session import UserSession
    from app.models.processing_queue import ProcessingQueue

    # Create all tables
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables created successfully")


def get_database() -> Generator[Session, None, None]:
    """
    Database dependency for FastAPI.
    Provides a database session for each request.
    """
    if SessionLocal is None:
        create_database_engine()

    db = SessionLocal()
    try:
        yield db
    except Exception as e:
        logger.error(f"Database session error: {e}")
        db.rollback()
        raise
    finally:
        db.close()


def close_database() -> None:
    """Close database connections."""
    global engine, SessionLocal

    if engine:
        engine.dispose()
        logger.info("Database connections closed")


class DatabaseManager:
    """High-level database management class."""

    def __init__(self):
        self.engine = None
        self.SessionLocal = None
        self._initialize()

    def _initialize(self) -> None:
        """Initialize database connection."""
        settings = get_settings()
        self.engine = create_engine(
            settings.database_url,
            connect_args={"check_same_thread": False}
            if settings.database_url.startswith("sqlite")
            else {},
            poolclass=StaticPool
            if settings.database_url.startswith("sqlite")
            else None,
        )
        self.SessionLocal = sessionmaker(
            autocommit=False, autoflush=False, bind=self.engine
        )

    def create_tables(self) -> None:
        """Create all database tables."""
        Base.metadata.create_all(bind=self.engine)

    def get_session(self) -> Session:
        """Get a new database session."""
        return self.SessionLocal()

    def health_check(self) -> bool:
        """Check if database connection is healthy."""
        try:
            session = self.get_session()
            session.execute("SELECT 1")
            session.close()
            return True
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False


# Global database manager instance
db_manager = DatabaseManager()


def get_db_manager() -> DatabaseManager:
    """Get the global database manager instance."""
    return db_manager


# Utility functions for common database operations
def execute_raw_sql(sql: str, params: dict = None) -> list:
    """Execute raw SQL query and return results."""
    session = db_manager.get_session()
    try:
        result = session.execute(sql, params or {})
        return result.fetchall()
    finally:
        session.close()


def backup_database(backup_path: str) -> bool:
    """Create a backup of the SQLite database."""
    if not get_settings().database_url.startswith("sqlite"):
        logger.warning("Database backup only supported for SQLite")
        return False

    try:
        import shutil

        db_path = get_settings().database_url.replace("sqlite:///", "")
        shutil.copy2(db_path, backup_path)
        logger.info(f"Database backed up to: {backup_path}")
        return True
    except Exception as e:
        logger.error(f"Database backup failed: {e}")
        return False


def restore_database(backup_path: str) -> bool:
    """Restore database from backup."""
    if not get_settings().database_url.startswith("sqlite"):
        logger.warning("Database restore only supported for SQLite")
        return False

    try:
        import shutil

        db_path = get_settings().database_url.replace("sqlite:///", "")
        shutil.copy2(backup_path, db_path)
        logger.info(f"Database restored from: {backup_path}")
        return True
    except Exception as e:
        logger.error(f"Database restore failed: {e}")
        return False


# Database initialization should be called explicitly, not on import
# if not get_settings().test_mode:
#     init_database()
