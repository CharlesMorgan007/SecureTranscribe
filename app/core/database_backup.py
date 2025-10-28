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

    logger.info("DEBUG: Starting create_database_engine function")
    settings = get_settings()
    database_url = settings.database_url
    logger.info(f"DEBUG: Database URL: {database_url}")

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
    logger.info(f"DEBUG: Engine type: {type(engine)}")
    logger.info(f"DEBUG: Engine is None: {engine is None}")
    logger.info(f"DEBUG: SessionLocal is None: {SessionLocal is None}")

    # Verify engine was created successfully
    if engine is None:
        logger.error("DEBUG: Engine is None after creation!")
        raise RuntimeError("Failed to create database engine")
    if SessionLocal is None:
        logger.error("DEBUG: SessionLocal is None after creation!")
        raise RuntimeError("Failed to create SessionLocal")


def init_database() -> None:
    """Initialize database tables and create required directories."""
    global engine, SessionLocal

    # Simple, direct approach that works
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

    # Create tables with proper error handling
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        if "tuple index out of range" in str(e) or "no such table" in str(e).lower():
            logger.warning("Database schema inconsistency detected, performing complete reset...")
            # Auto-reset --> database
            reset_database()
        else:
            logger.error(f"Database initialization failed: {e}")
            raise RuntimeError(f"Database initialization failed: {e}")

    logger.info(f"Database initialization complete, engine: {engine is not None}, SessionLocal: {SessionLocal is not None}")
    )


def check_and_repair_database() -> bool:
    """
    Check database integrity and repair if needed.
    Returns True if repair was performed.
    """
    global engine

    try:
        # Try to access a sample UserSession row
        logger.info(f"DEBUG: check_and_repair_database, engine: {engine is not None}")
        if engine is None:
            logger.warning("Database engine not available for integrity check")
            return False

        with engine.connect() as conn:
            result = conn.execute("SELECT COUNT(*) FROM user_sessions LIMIT 1")
            result.fetchone()
        return False
    except Exception as e:
        if "tuple index out of range" in str(e) or "no such table" in str(e).lower():
            logger.warning("Database corruption detected, performing repair...")
            try:
                # Backup existing data if possible
                backup_db_file = (
                    f"securetranscribe_backup_{int(datetime.utcnow().timestamp())}.db"
                )
                import shutil
                import os

                if os.path.exists("securetranscribe.db"):
                    shutil.copy2("securetranscribe.db", backup_db_file)
                    logger.info(f"Created database backup: {backup_db_file}")

                # Reset database
                engine.dispose()
                Base.metadata.drop_all(bind=engine)
                Base.metadata.create_all(bind=engine)
                logger.info("Database has been reset successfully")

                # Re-create engine after reset
                logger.info("DEBUG: Re-creating database engine...")
                create_database_engine()
                logger.info(f"DEBUG: After re-creation, engine: {engine is not None}")

                return True
            except Exception as e:
                logger.error(f"Failed to repair database: {repair_error}")
                return False


def reset_database() -> None:
    """
    Complete database reset to resolve schema corruption issues.
    This function should be called when 'tuple index out of range' error occurs.
    """
    global engine, SessionLocal

    logger.info("Performing complete database reset...")

    try:
        # Backup existing database if possible
        import shutil
        import os
        import datetime

        db_path = "securetranscribe.db"
        if os.path.exists(db_path):
            backup_path = (
                f"securetranscribe_backup_{int(datetime.datetime.now().timestamp())}.db"
            )
            shutil.copy2(db_path, backup_path)
            logger.info(f"Created database backup: {backup_path}")

        # Remove existing database file
        if os.path.exists(db_path):
            os.remove(db_path)
            logger.info("Removed corrupted database file")

        # Dispose existing engine
        if engine:
            engine.dispose()
            logger.info("Disposed existing engine")

        # Recreate engine and database
        create_database_engine()

        # Force model registration
        from app.models import (
            Speaker,
            Transcription,
            UserSession,
            ProcessingQueue,
        )

        # Create fresh database schema
        Base.metadata.create_all(bind=engine)
        logger.info("Created fresh database with correct schema")

        logger.info("Database reset completed successfully")

    except Exception as e:
        logger.error(f"Database reset failed: {e}")
        raise RuntimeError(f"Failed to reset database: {e}")


def get_database() -> Generator[Session, None, None]:
    """
    Database dependency for FastAPI.
    Provides a database session for each request.
    """
    global SessionLocal, engine

    logger.info(
        f"DEBUG: get_database called, engine: {engine is not None}, SessionLocal: {SessionLocal is not None}"
    )

    if SessionLocal is None or engine is None:
        logger.info("DEBUG: Creating database engine in get_database()")
        create_database_engine()
        logger.info(
            f"DEBUG: After create_database_engine, engine: {engine is not None}, SessionLocal: {SessionLocal is not None}"
        )

        if SessionLocal is None or engine is None:
            logger.error("DEBUG: Database components still not properly initialized")
            raise RuntimeError("Database components not properly initialized")

    # Update global variables for backward compatibility
    globals()["engine"] = engine
    globals()["SessionLocal"] = SessionLocal

    db = SessionLocal()
    try:
        yield db
    except Exception as e:
        # Auto-reset database on schema corruption errors
        if "tuple index out of range" in str(e):
            logger.error(f"Database schema corruption detected: {e}")
            reset_database()
            raise RuntimeError(
                "Database was reset due to schema corruption. Please retry the request."
            )

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


def get_database_manager() -> DatabaseManager:
    """Get database manager for external use."""
    return db_manager


def get_db_manager() -> DatabaseManager:
    """Get the global database manager instance."""
    return db_manager


def refresh_database_sessions():
    """Refresh all database sessions to clear cache."""
    if SessionLocal:
        SessionLocal.remove()


# Utility functions for common database operations
def execute_raw_sql(sql: str, params: dict = None) -> list:
    """Execute raw SQL query and return results."""
    global SessionLocal, engine

    session = SessionLocal()
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
