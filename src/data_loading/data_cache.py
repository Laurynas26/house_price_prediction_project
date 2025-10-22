import hashlib
import json
import pickle
from pathlib import Path
from datetime import datetime
from typing import Any, Optional, Union

import pandas as pd


class CacheManager:
    """
    General-purpose cache manager for storing and loading intermediate
    pipeline data (e.g., preprocessed or feature-engineered datasets).

    Adapted for nested YAML configs:
    - Can hash specific sub-configs (e.g., per-model or feature-exp block)
    - Automatically handles deeply nested dictionaries
    """

    def __init__(
        self,
        cache_dir: Union[str, Path] = "data/cache",
        use_timestamp: bool = False,
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.use_timestamp = use_timestamp

    # -------------------------------------------------------------------------
    # Internal Helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def _normalize_config(
        config: Optional[dict], scope: Optional[str] = None
    ) -> Optional[dict]:
        """
        Extract a sub-config by scope name (e.g., 'preprocessing' or 'model').
        If scope is not found, returns the full config instead of raising KeyError.
        """
        if config is None:
            return None

        if scope:
            if not isinstance(config, dict):
                raise TypeError(
                    f"Expected dict for config, got {type(config)}"
                )
            return config.get(
                scope, config
            )  # <- return subconfig or full config

        return config

    @staticmethod
    def _make_hash(config: Optional[dict]) -> Optional[str]:
        """Generate a short deterministic hash string from a nested config dict."""
        if config is None:
            return None
        # Ensure consistent key order and hash only relevant subset
        config_str = json.dumps(config, sort_keys=True, default=str)
        return hashlib.md5(config_str.encode("utf-8")).hexdigest()[:8]

    def _get_path(
        self, name: str, hash_key: Optional[str] = None, suffix: str = ".pkl"
    ) -> Path:
        """Construct the cache file path."""
        parts = [name]
        if hash_key:
            parts.append(hash_key)
        if self.use_timestamp:
            parts.append(datetime.now().strftime("%Y%m%d_%H%M%S"))
        return self.cache_dir / f"{'_'.join(parts)}{suffix}"

    def _resolve_latest_path(
        self, name: str, hash_key: Optional[str]
    ) -> Optional[Path]:
        """Find the most recent cache file for the given name and hash."""
        pattern = f"{name}_{hash_key}*" if hash_key else f"{name}*"
        matches = sorted(self.cache_dir.glob(f"{pattern}.pkl"))
        return matches[-1] if matches else None

    # -------------------------------------------------------------------------
    # Core API
    # -------------------------------------------------------------------------

    def exists(
        self,
        name: str,
        config: Optional[dict] = None,
        scope: Optional[str] = None,
    ) -> bool:
        """Check if cached data exists for the given name and config subset."""
        subconfig = self._normalize_config(config, scope)
        hash_key = self._make_hash(subconfig)
        return any(self.cache_dir.glob(f"{name}_{hash_key}*.pkl"))

    def save(
        self,
        obj: Any,
        name: str,
        config: Optional[dict] = None,
        scope: Optional[str] = None,
        as_pickle: bool = True,
    ) -> Path:
        """Save a Python object or DataFrame with config-based versioning."""
        subconfig = self._normalize_config(config, scope)
        hash_key = self._make_hash(subconfig)
        path = self._get_path(name, hash_key)

        if as_pickle or not hasattr(obj, "to_pickle"):
            with open(path, "wb") as f:
                pickle.dump(obj, f)
        else:
            obj.to_pickle(path)

        print(f"💾 Cached [{name}] → {path}")
        return path

    def load(
        self,
        name: str,
        config: Optional[dict] = None,
        scope: Optional[str] = None,
    ) -> Any:
        """Load the most recent cache matching the given name and config subset."""
        subconfig = self._normalize_config(config, scope)
        hash_key = self._make_hash(subconfig)
        path = self._resolve_latest_path(name, hash_key)

        if not path or not path.exists():
            raise FileNotFoundError(
                f"No cache found for: {name} ({scope or 'full'})"
            )

        with open(path, "rb") as f:
            obj = pickle.load(f)

        print(f"✅ Loaded [{name}] ← {path.name}")
        return obj

    def clear(self, name: Optional[str] = None):
        """Remove one or all cached items."""
        if name:
            for file in self.cache_dir.glob(f"{name}*"):
                file.unlink(missing_ok=True)
            print(f"🧹 Cleared cache for: {name}")
        else:
            for file in self.cache_dir.glob("*"):
                file.unlink()
            print("🧹 Cleared all caches.")
