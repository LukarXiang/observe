from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    root: Path = Path(".")
    data_dir: Path = Path("data")
    runs_dir: Path = Path("runs")
    initial_cash: float = 1_000_000.0
    annualization: int = 242
    rebalance_every: int = 5

    @classmethod
    def from_root(cls, root):
        root = Path(root)
        return cls(root = root, data_dir = root / "data", runs_dir = root / "runs")
