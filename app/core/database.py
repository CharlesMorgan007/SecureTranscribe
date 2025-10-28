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
from app.utils.exceptions import SecureTranscribeError
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

    # Verify engine was created successfully
    if engine is None:
        raise RuntimeError("Failed to create database engine")
    if SessionLocal is None:
        raise RuntimeError("Failed to create SessionLocal")


def init_database() -> None:
    """Initialize database tables and create required directories."""
    global engine, SessionLocal

    # Create engine if needed
    if engine is None:
        create_database_engine()

    if engine is None:
        raise RuntimeError("Failed to create database engine")

    # Import all models to ensure they're registered with Base
    from app.models import (
        Speaker,
        Transcription,
        UserSession,
        ProcessingQueue,
    )

    # Create tables with error handling
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully")
    except Exception as e:
        if "tuple index out of range" in str(e) or "no such table" in str(e).lower():
            logger.warning(
                "Database schema inconsistency detected, performing complete reset..."
            )
            try:
                # Backup existing database if possible
                import shutil
                import datetime

                db_path = "securetranscribe.db"
                if os.path.exists(db_path):
                    backup_path = f"securetranscribe_backup_{int(datetime.datetime.now().timestamp())}.db"
                    shutil.copy2(db_path, backup_path)
                    logger.info(f"Created database backup: {backup_path}")

                # Dispose and recreate database
                engine.dispose()
                Base.metadata.drop_all(bind=engine)
                Base.metadata.create_all(bind=engine)
                logger.info("Database reset completed successfully")
            except Exception as repair_error:
                logger.error(f"Database repair failed: {repair_error}")
                raise RuntimeError(f"Failed to repair database: {repair_error}")
        else:
            logger.error(f"Database initialization failed: {e}")
            raise RuntimeError(f"Failed to initialize database: {e}")


def get_database() -> Generator[Session, None, None]:
    """
    Database dependency for FastAPI.
    Provides a database session for each request.
    """
    global SessionLocal, engine

    if SessionLocal is None or engine is None:
        create_database_engine()

    if SessionLocal is None or engine is None:
        raise RuntimeError("Database components not properly initialized")

    db = SessionLocal()
    try:
        yield db
    except Exception as e:
        logger.error(f"Database session error: {e}")
        db.rollback()
        # Auto-reset database on schema corruption errors
        if "tuple index out of range" in str(e):
            logger.error(f"Database schema corruption detected: {e}")
            init_database()
            raise RuntimeError(
                "Database was reset due to schema corruption. Please retry request."
            )
        raise


def close_database() -> None:
    """Close database connections."""
    global engine, SessionLocal

    if engine:
        engine.dispose()
        logger.info("Database connections closed")


# Utility functions for common database operations
def execute_raw_sql(sql: str, params: dict = None) -> list:
    """Execute raw SQL query and return results."""
    global SessionLocal, engine

    if SessionLocal is None or engine is None:
        create_database_engine()

    session = SessionLocal()
    try:
        result = session.execute(sql, params or {})
        return result.fetchall()
    finally:
        session.close()


def backup_database(backup_path: str) -> bool:
    """Create a backup of the SQLite database."""
    global engine

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
    global engine

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


# Simple database manager for easier access
class DatabaseManager:
    """High-level database management class."""

    def __init__(self):
        self.engine = None
        self.SessionLocal = None
        self._initialize()

    def _initialize(self):
        """Initialize database connection."""
        global engine, SessionLocal
        create_database_engine()
        self.engine = engine
        self.SessionLocal = SessionLocal

    def create_tables(self):
        """Create all database tables."""
        global engine
        if engine:
            Base.metadata.create_all(bind=engine)
            logger.info("Database tables created successfully")

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


def get_database_manager() -> DatabaseManager:
    """Get database manager for external use."""
    return db_manager


def get_db_manager() -> DatabaseManager:
    """Get the global database manager instance."""
    return db_manager


def refresh_database_sessions():
    """Refresh all database sessions to clear cache."""
    global SessionLocal
    if SessionLocal:
        SessionLocal.remove()
