"""Dataclass for clip"""
from dataclasses import dataclass
from pathlib import Path
from typing import List

@dataclass
class HashLabels:
    # filemap: Path
    # """filename to load sam features map tensor"""
    filenames_labels: List[Path]
    """filenames to load sam features tensor"""