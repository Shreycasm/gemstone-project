from dataclasses import is_dataclass, asdict
import json
import yaml
import joblib
from pathlib import Path
from typing import Any, Union
import pandas as pd
import urllib.parse


def _ensure_path(file_path: Union[str, Path]) -> Path:
    return Path(file_path)


def load_data(file_path: Union[str, Path]) -> pd.DataFrame:
    path_or_url = str(file_path)
    parsed = urllib.parse.urlparse(path_or_url)
    is_url = parsed.scheme in ("http", "https")

    if is_url:
        return pd.read_csv(path_or_url, encoding="utf-8", low_memory=False)
    else:
        p = _ensure_path(path_or_url)
        suffix = p.suffix.lstrip(".").lower()
        if suffix == "csv":
            return pd.read_csv(p, encoding="utf-8", low_memory=False)
        elif suffix in ("parquet", "pq"):
            return pd.read_parquet(p)
        else:
            raise ValueError(f"Unsupported extension: {suffix} for file {p}")


def save_data(df: pd.DataFrame, file_path: Union[str, Path]) -> None:
    p = _ensure_path(file_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    suffix = p.suffix.lstrip(".").lower()

    if suffix == "csv":
        df.to_csv(p, index=False, encoding="utf-8")
    elif suffix in ("parquet", "pq"):
        df.to_parquet(p, index=False, engine="pyarrow")
    else:
        raise ValueError(f"Unsupported extension: {suffix} for file {p}")


def save_joblib_object(file_path: Union[str, Path], obj: Any) -> None:
    p = _ensure_path(file_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "wb") as f:
        joblib.dump(obj, f)



def load_joblib_object(file_path: Union[str, Path]) -> Any:
    p = _ensure_path(file_path)
    return joblib.load(p)


def load_yaml_file(file_path: Union[str, Path]) -> dict:
    p = _ensure_path(file_path)
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_json_file(file_path: Union[str, Path], data: Any, indent: int = 4) -> None:
    p = _ensure_path(file_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    if is_dataclass(data):
        data = asdict(data)

    def default_serializer(obj):
        if isinstance(obj, Path):
            return str(obj)
        return str(obj)

    if not p.suffix:
        p = p.with_suffix(".json")

    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, default=default_serializer, ensure_ascii=False)


def load_json_file(file_path: Union[str, Path]) -> dict:
    p = _ensure_path(file_path)
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)
