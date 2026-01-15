import hashlib
import json
import os
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Union

import boto3
import botocore


class CacheManager:
    """
    General-purpose cache manager for storing and loading intermediate
    pipeline data (e.g., preprocessed or feature-engineered datasets).

    Features:
    - Nested YAML config hashing for scoped cache
    - Automatic upload to S3
    - Auto-download from S3 if cache not present locally
    """

    def __init__(
        self,
        cache_dir: Union[str, Path] = "data/cache",
        use_timestamp: bool = False,
    ):
        self.s3_bucket = os.environ.get("CACHE_BUCKET")
        self.s3_prefix = os.environ.get("CACHE_PREFIX", "cache/")
        self.s3 = boto3.client("s3") if self.s3_bucket else None

        self.is_lambda = bool(os.environ.get("AWS_LAMBDA_FUNCTION_NAME"))

        # Writable cache directory
        if self.is_lambda:
            self.cache_dir = Path("/tmp/cache")
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            prepackaged_path = Path(cache_dir)
            self.prepackaged_cache_dir = (
                prepackaged_path if prepackaged_path.exists() else None
            )
        else:
            self.cache_dir = Path(cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self.prepackaged_cache_dir = None

        self.use_timestamp = use_timestamp

    # ----------------------- Helpers -----------------------

    @staticmethod
    def _normalize_config(
        config: Optional[dict], scope: Optional[str] = None
    ) -> Optional[dict]:
        if config is None:
            return None
        if scope:
            if not isinstance(config, dict):
                raise TypeError(
                    f"Expected dict for config, got {type(config)}"
                )
            return config.get(scope, config)
        return config

    @staticmethod
    def _make_hash(config: Optional[dict]) -> Optional[str]:
        if config is None:
            return None
        config_str = json.dumps(config, sort_keys=True, default=str)
        return hashlib.md5(config_str.encode("utf-8")).hexdigest()[:8]

    def _get_path(
        self, name: str, hash_key: Optional[str] = None, suffix: str = ".pkl"
    ) -> Path:
        parts = [name]
        if hash_key:
            parts.append(hash_key)
        if self.use_timestamp:
            parts.append(datetime.now().strftime("%Y%m%d_%H%M%S"))
        return self.cache_dir / f"{'_'.join(parts)}{suffix}"

    def _resolve_latest_path(
        self, name: str, hash_key: Optional[str]
    ) -> Optional[Path]:
        pattern = f"{name}_{hash_key}*.pkl"
        local_matches = sorted(self.cache_dir.glob(pattern))
        if local_matches:
            return local_matches[-1]
        if self.is_lambda and self.prepackaged_cache_dir:
            container_matches = sorted(
                self.prepackaged_cache_dir.glob(pattern)
            )
            if container_matches:
                return container_matches[-1]
        fallback = self.cache_dir / f"{name}_{hash_key}.pkl"
        return fallback if fallback.exists() else None

    def _s3_exists(self, key: str) -> bool:
        if not self.s3:
            return False
        try:
            self.s3.head_object(Bucket=self.s3_bucket, Key=key)
            return True
        except botocore.exceptions.ClientError:
            return False

    def _download_from_s3(self, key: str, local_path: Path):
        local_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"⬇️ Downloading from S3: s3://{self.s3_bucket}/{key}")
        self.s3.download_file(self.s3_bucket, key, str(local_path))
        return local_path

    def _upload_to_s3(self, local_path: Path, key: str):
        if not self.s3:
            return
        print(f"⬆️ Uploading to S3: s3://{self.s3_bucket}/{key}")
        self.s3.upload_file(str(local_path), self.s3_bucket, key)

    # ----------------------- Core API -----------------------

    def exists(
        self,
        name: str,
        config: Optional[dict] = None,
        scope: Optional[str] = None,
    ) -> bool:
        subconfig = self._normalize_config(config, scope)
        hash_key = self._make_hash(subconfig)
        pattern = f"{name}_{hash_key}.pkl" if hash_key else f"{name}.pkl"

        # Check local cache
        if any(self.cache_dir.glob(pattern)):
            return True

        # Check prepackaged container cache
        if self.is_lambda and self.prepackaged_cache_dir:
            if any(self.prepackaged_cache_dir.glob(pattern)):
                return True

        # Check S3
        if self.s3:
            s3_key = f"{self.s3_prefix}{name}.pkl"
            try:
                self.s3.head_object(Bucket=self.s3_bucket, Key=s3_key)
                return True
            except self.s3.exceptions.ClientError:
                return False

        return False

    def save(
        self,
        obj: Any,
        name: str,
        config: Optional[dict] = None,
        scope: Optional[str] = None,
        as_pickle: bool = True,
    ) -> Path:
        subconfig = self._normalize_config(config, scope)
        hash_key = self._make_hash(subconfig)
        path = self._get_path(name, hash_key)

        if as_pickle or not hasattr(obj, "to_pickle"):
            with open(path, "wb") as f:
                pickle.dump(obj, f)
        else:
            obj.to_pickle(path)

        # Upload to S3 automatically
        if self.s3_bucket:
            s3_key = f"{self.s3_prefix}{path.name}"
            self._upload_to_s3(path, s3_key)

        print(f"💾 Cached [{name}] → {path}")
        return path

    def load(
        self,
        name: str,
        config: Optional[dict] = None,
        scope: Optional[str] = None,
    ) -> Any:
        subconfig = self._normalize_config(config, scope)
        hash_key = self._make_hash(subconfig)
        filename = f"{name}_{hash_key}.pkl" if hash_key else f"{name}.pkl"

        # Try local or container first
        path = self._resolve_latest_path(name, hash_key)

        # If not found locally, try S3
        if (not path or not path.exists()) and self.s3:
            s3_key = f"{self.s3_prefix}{filename}"
            if self._s3_exists(s3_key):
                path = self.cache_dir / filename
                self._download_from_s3(s3_key, path)

        if not path or not path.exists():
            raise FileNotFoundError(f"No cache found for: {name}")

        with open(path, "rb") as f:
            obj = pickle.load(f)
        return obj

    def clear(self, name: Optional[str] = None):
        if name:
            for file in self.cache_dir.glob(f"{name}*"):
                file.unlink(missing_ok=True)
            print(f"🧹 Cleared cache for: {name}")
        else:
            for file in self.cache_dir.glob("*"):
                file.unlink()
            print("🧹 Cleared all caches.")
