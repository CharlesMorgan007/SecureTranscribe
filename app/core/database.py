"""
Database setup and connection management for SecureTranscribe.
Handles SQLite database initialization, connection pooling, and session management.
"""

import os
import logging
import time
import threading
from contextlib import contextmanager
from sqlalchemy import create_engine, event
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import NullPool
from sqlalchemy.exc import OperationalError
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

    # SQLite-specific configuration with thread safety
    if database_url.startswith("sqlite"):
        engine = create_engine(
            database_url,
            connect_args={
                "check_same_thread": False,
                "timeout": 30,
            },
            poolclass=NullPool,
            echo=DATABASE_SETTINGS["echo"],
            pool_pre_ping=DATABASE_SETTINGS["pool_pre_ping"],
            pool_recycle=DATABASE_SETTINGS["pool_recycle"],
        )

        # Enable WAL mode for better concurrent access
        @event.listens_for(engine, "connect")
        def set_sqlite_pragma(dbapi_connection, connection_record):
            cursor = dbapi_connection.cursor()
            # Enable Write-Ahead Logging for better concurrency
            cursor.execute("PRAGMA journal_mode=WAL")
            # Set busy timeout to handle locking
            cursor.execute("PRAGMA busy_timeout=30000")  # 30 seconds
            # Enable foreign key constraints
            cursor.execute("PRAGMA foreign_keys=ON")
            # Optimize for concurrent access
            cursor.execute("PRAGMA synchronous=NORMAL")
            cursor.execute("PRAGMA cache_size=10000")
            cursor.execute("PRAGMA temp_store=memory")
            cursor.close()
    else:
        # PostgreSQL/MySQL configuration
        engine = create_engine(
            database_url,
            echo=DATABASE_SETTINGS["echo"],
            pool_pre_ping=DATABASE_SETTINGS["pool_pre_ping"],
            pool_recycle=DATABASE_SETTINGS["pool_recycle"],
        )

    SessionLocal = sessionmaker(
        autocommit=False, autoflush=False, expire_on_commit=False, bind=engine
    )
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
        Transcription,
        UserSession,
        ProcessingQueue,
    )

    # Create tables with error handling
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully")
    except Exception as e:
        if any(
            m in str(e).lower()
            for m in ("database disk image is malformed", "file is not a database")
        ):
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
    Provides a database session for each request with proper error handling.
    """
    global SessionLocal, engine

    if SessionLocal is None or engine is None:
        create_database_engine()

    if SessionLocal is None or engine is None:
        raise RuntimeError("Database components not properly initialized")

    db = SessionLocal()
    retry_count = 0
    max_retries = 3

    while retry_count < max_retries:
        try:
            yield db
            # Success - exit retry loop
            break
        except OperationalError as e:
            if "database is locked" in str(e).lower() and retry_count < max_retries - 1:
                retry_count += 1
                logger.warning(
                    f"Database locked, retrying ({retry_count}/{max_retries})"
                )
                time.sleep(0.5 * retry_count)  # Exponential backoff
                try:
                    db.rollback()
                finally:
                    db.close()
                db = SessionLocal()
                continue
            else:
                logger.error(f"Database operational error: {e}")
                db.rollback()
                raise
        except Exception as e:
            logger.error(f"Database session error: {e}")
            db.rollback()
            # Only reset database on critical schema corruption errors
            if (
                any(
                    m in str(e).lower()
                    for m in (
                        "database disk image is malformed",
                        "file is not a database",
                    )
                )
                and retry_count == 0
            ):
                logger.error(f"Database schema corruption detected: {e}")
                # Try to backup before reset
                try:
                    backup_database(
                        f"securetranscribe_corruption_backup_{int(time.time())}.db"
                    )
                except Exception as backup_error:
                    logger.error(f"Failed to backup corrupted database: {backup_error}")

                init_database()
                raise RuntimeError(
                    "Database was reset due to schema corruption. Please retry request."
                )
            raise
        finally:
            db.close()


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


# Thread-local storage for database sessions
_thread_local = threading.local()


# Simple database manager for easier access
class DatabaseManager:
    """High-level database management class with thread safety."""

    def __init__(self):
        self.engine = None
        self.SessionLocal = None
        self._initialize()
        self._lock = threading.RLock()  # Reentrant lock for write operations

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

    @contextmanager
    def get_thread_session(self):
        """Get a thread-local database session with automatic cleanup."""
        if not hasattr(_thread_local, "session"):
            _thread_local.session = self.SessionLocal()

        session = _thread_local.session
        try:
            yield session
        except Exception as e:
            session.rollback()
            logger.error(f"Thread session error: {e}")
            raise
        finally:
            # Don't close thread-local sessions here, let them be reused
            pass

    def get_session(self) -> Session:
        """Get a new database session."""
        return self.SessionLocal()

    def health_check(self) -> bool:
        """Check if database connection is healthy."""
        try:
            session = self.get_session()
            from sqlalchemy import text

            session.execute(text("SELECT 1"))
            session.close()
            return True
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False

    @contextmanager
    def write_lock(self):
        """Context manager for write operations to ensure thread safety."""
        self._lock.acquire()
        try:
            yield
        finally:
            self._lock.release()

    def execute_with_retry(self, operation, max_retries=3):
        """Execute a database operation with retry logic for locking issues."""
        for attempt in range(max_retries):
            try:
                return operation()
            except OperationalError as e:
                if "database is locked" in str(e).lower() and attempt < max_retries - 1:
                    logger.warning(
                        f"Database locked, retrying operation (attempt {attempt + 1})"
                    )
                    time.sleep(0.1 * (attempt + 1))  # Exponential backoff
                    continue
                else:
                    raise
            except Exception as e:
                logger.error(f"Database operation failed: {e}")
                raise


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
