"""Prediction caching utilities for efficient inference."""

import hashlib
import json
import pickle
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Any, Union
from dataclasses import dataclass, asdict
import numpy as np
import threading
import logging

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Container for a cache entry."""

    key: str
    predictions: np.ndarray
    model_name: str
    model_version: str
    created_at: str
    expires_at: Optional[str] = None
    metadata: Optional[dict] = None


class PredictionCache:
    """
    Cache for molecular property predictions.

    Supports both in-memory and persistent (SQLite) caching with
    configurable TTL and cache size limits.

    Args:
        cache_dir: Directory for persistent cache (None for memory-only)
        max_size: Maximum number of entries in memory cache
        ttl_hours: Time-to-live in hours (None for no expiry)
        use_sqlite: Whether to use SQLite for persistence
    """

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        max_size: int = 10000,
        ttl_hours: Optional[float] = 24.0,
        use_sqlite: bool = True,
    ):
        self.max_size = max_size
        self.ttl_hours = ttl_hours
        self.use_sqlite = use_sqlite

        # In-memory cache
        self._memory_cache: dict[str, CacheEntry] = {}
        self._access_order: list[str] = []  # LRU tracking
        self._lock = threading.Lock()

        # Persistent cache
        if cache_dir and use_sqlite:
            self.cache_dir = Path(cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self.db_path = self.cache_dir / "predictions.db"
            self._init_db()
        else:
            self.cache_dir = None
            self.db_path = None

        # Statistics
        self.hits = 0
        self.misses = 0

    def _init_db(self) -> None:
        """Initialize SQLite database."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                key TEXT PRIMARY KEY,
                predictions BLOB,
                model_name TEXT,
                model_version TEXT,
                created_at TEXT,
                expires_at TEXT,
                metadata TEXT
            )
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_expires
            ON predictions(expires_at)
        """)
        conn.commit()
        conn.close()

    def _generate_key(
        self,
        smiles: str,
        model_name: str,
        model_version: str,
    ) -> str:
        """Generate cache key from SMILES and model info."""
        key_str = f"{smiles}:{model_name}:{model_version}"
        return hashlib.sha256(key_str.encode()).hexdigest()

    def _is_expired(self, entry: CacheEntry) -> bool:
        """Check if cache entry is expired."""
        if entry.expires_at is None:
            return False
        expires = datetime.fromisoformat(entry.expires_at)
        return datetime.now() > expires

    def get(
        self,
        smiles: str,
        model_name: str,
        model_version: str,
    ) -> Optional[np.ndarray]:
        """
        Get cached prediction for a molecule.

        Args:
            smiles: SMILES string
            model_name: Model name
            model_version: Model version

        Returns:
            Cached predictions or None if not found
        """
        key = self._generate_key(smiles, model_name, model_version)

        # Check memory cache first
        with self._lock:
            if key in self._memory_cache:
                entry = self._memory_cache[key]
                if not self._is_expired(entry):
                    self.hits += 1
                    # Update LRU order
                    if key in self._access_order:
                        self._access_order.remove(key)
                    self._access_order.append(key)
                    return entry.predictions
                else:
                    # Remove expired entry
                    del self._memory_cache[key]
                    if key in self._access_order:
                        self._access_order.remove(key)

        # Check SQLite cache
        if self.db_path:
            entry = self._get_from_db(key)
            if entry and not self._is_expired(entry):
                self.hits += 1
                # Populate memory cache
                with self._lock:
                    self._memory_cache[key] = entry
                    self._access_order.append(key)
                    self._enforce_size_limit()
                return entry.predictions

        self.misses += 1
        return None

    def set(
        self,
        smiles: str,
        model_name: str,
        model_version: str,
        predictions: np.ndarray,
        metadata: Optional[dict] = None,
    ) -> None:
        """
        Cache a prediction.

        Args:
            smiles: SMILES string
            model_name: Model name
            model_version: Model version
            predictions: Prediction array
            metadata: Optional metadata
        """
        key = self._generate_key(smiles, model_name, model_version)

        now = datetime.now()
        expires_at = None
        if self.ttl_hours:
            expires_at = (now + timedelta(hours=self.ttl_hours)).isoformat()

        entry = CacheEntry(
            key=key,
            predictions=predictions,
            model_name=model_name,
            model_version=model_version,
            created_at=now.isoformat(),
            expires_at=expires_at,
            metadata=metadata,
        )

        # Update memory cache
        with self._lock:
            self._memory_cache[key] = entry
            if key in self._access_order:
                self._access_order.remove(key)
            self._access_order.append(key)
            self._enforce_size_limit()

        # Update SQLite cache
        if self.db_path:
            self._save_to_db(entry)

    def get_batch(
        self,
        smiles_list: list[str],
        model_name: str,
        model_version: str,
    ) -> tuple[dict[str, np.ndarray], list[str]]:
        """
        Get cached predictions for a batch of molecules.

        Args:
            smiles_list: List of SMILES strings
            model_name: Model name
            model_version: Model version

        Returns:
            Tuple of (cached_predictions dict, uncached_smiles list)
        """
        cached = {}
        uncached = []

        for smiles in smiles_list:
            pred = self.get(smiles, model_name, model_version)
            if pred is not None:
                cached[smiles] = pred
            else:
                uncached.append(smiles)

        return cached, uncached

    def set_batch(
        self,
        smiles_list: list[str],
        predictions: np.ndarray,
        model_name: str,
        model_version: str,
    ) -> None:
        """
        Cache predictions for a batch of molecules.

        Args:
            smiles_list: List of SMILES strings
            predictions: Predictions array (N, num_tasks)
            model_name: Model name
            model_version: Model version
        """
        for i, smiles in enumerate(smiles_list):
            self.set(smiles, model_name, model_version, predictions[i])

    def _enforce_size_limit(self) -> None:
        """Remove oldest entries if cache exceeds size limit."""
        while len(self._memory_cache) > self.max_size:
            if self._access_order:
                oldest_key = self._access_order.pop(0)
                if oldest_key in self._memory_cache:
                    del self._memory_cache[oldest_key]

    def _get_from_db(self, key: str) -> Optional[CacheEntry]:
        """Get entry from SQLite database."""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM predictions WHERE key = ?",
                (key,)
            )
            row = cursor.fetchone()
            conn.close()

            if row:
                return CacheEntry(
                    key=row[0],
                    predictions=pickle.loads(row[1]),
                    model_name=row[2],
                    model_version=row[3],
                    created_at=row[4],
                    expires_at=row[5],
                    metadata=json.loads(row[6]) if row[6] else None,
                )
        except Exception as e:
            logger.error(f"Error reading from cache DB: {e}")

        return None

    def _save_to_db(self, entry: CacheEntry) -> None:
        """Save entry to SQLite database."""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO predictions
                (key, predictions, model_name, model_version, created_at, expires_at, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    entry.key,
                    pickle.dumps(entry.predictions),
                    entry.model_name,
                    entry.model_version,
                    entry.created_at,
                    entry.expires_at,
                    json.dumps(entry.metadata) if entry.metadata else None,
                )
            )
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Error writing to cache DB: {e}")

    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self._memory_cache.clear()
            self._access_order.clear()

        if self.db_path:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            cursor.execute("DELETE FROM predictions")
            conn.commit()
            conn.close()

        logger.info("Cache cleared")

    def cleanup_expired(self) -> int:
        """
        Remove expired entries from cache.

        Returns:
            Number of entries removed
        """
        removed = 0

        # Memory cache
        with self._lock:
            expired_keys = [
                key for key, entry in self._memory_cache.items()
                if self._is_expired(entry)
            ]
            for key in expired_keys:
                del self._memory_cache[key]
                if key in self._access_order:
                    self._access_order.remove(key)
                removed += 1

        # SQLite cache
        if self.db_path:
            now = datetime.now().isoformat()
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            cursor.execute(
                "DELETE FROM predictions WHERE expires_at IS NOT NULL AND expires_at < ?",
                (now,)
            )
            removed += cursor.rowcount
            conn.commit()
            conn.close()

        logger.info(f"Removed {removed} expired cache entries")
        return removed

    def get_stats(self) -> dict:
        """Get cache statistics."""
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0.0

        return {
            "memory_size": len(self._memory_cache),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            "ttl_hours": self.ttl_hours,
        }


class FingerprintCache:
    """
    Cache for molecular fingerprints.

    Useful for speeding up repeated feature extraction.
    """

    def __init__(self, max_size: int = 50000):
        self.max_size = max_size
        self._cache: dict[str, np.ndarray] = {}
        self._access_order: list[str] = []
        self._lock = threading.Lock()

    def get(self, smiles: str) -> Optional[np.ndarray]:
        """Get cached fingerprint."""
        with self._lock:
            if smiles in self._cache:
                # Update LRU order
                if smiles in self._access_order:
                    self._access_order.remove(smiles)
                self._access_order.append(smiles)
                return self._cache[smiles].copy()
        return None

    def set(self, smiles: str, fingerprint: np.ndarray) -> None:
        """Cache a fingerprint."""
        with self._lock:
            self._cache[smiles] = fingerprint.copy()
            if smiles in self._access_order:
                self._access_order.remove(smiles)
            self._access_order.append(smiles)
            self._enforce_size_limit()

    def _enforce_size_limit(self) -> None:
        """Remove oldest entries if cache exceeds size limit."""
        while len(self._cache) > self.max_size:
            if self._access_order:
                oldest = self._access_order.pop(0)
                if oldest in self._cache:
                    del self._cache[oldest]

    def clear(self) -> None:
        """Clear cache."""
        with self._lock:
            self._cache.clear()
            self._access_order.clear()
