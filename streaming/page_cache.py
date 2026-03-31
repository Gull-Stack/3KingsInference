"""Page cache monitoring for expert streaming.

Tracks macOS page cache utilization to understand
expert residency and predict SSD read requirements.
"""

import subprocess
from dataclasses import dataclass


@dataclass
class PageCacheStats:
    """macOS page cache statistics."""
    pages_active: int = 0
    pages_inactive: int = 0
    pages_wired: int = 0
    pages_free: int = 0
    page_size: int = 16384  # Apple Silicon uses 16KB pages

    @property
    def active_gb(self) -> float:
        return (self.pages_active * self.page_size) / (1024**3)

    @property
    def inactive_gb(self) -> float:
        return (self.pages_inactive * self.page_size) / (1024**3)

    @property
    def file_cache_gb(self) -> float:
        """Estimated file-backed pages (page cache)."""
        return self.inactive_gb  # Rough approximation

    @property
    def free_gb(self) -> float:
        return (self.pages_free * self.page_size) / (1024**3)


def get_page_cache_stats() -> PageCacheStats:
    """Query macOS vm_stat for page cache information."""
    try:
        result = subprocess.run(
            ["vm_stat"], capture_output=True, text=True, timeout=5
        )
        stats = PageCacheStats()

        for line in result.stdout.split("\n"):
            if "Pages active" in line:
                stats.pages_active = int(line.split(":")[1].strip().rstrip("."))
            elif "Pages inactive" in line:
                stats.pages_inactive = int(line.split(":")[1].strip().rstrip("."))
            elif "Pages wired" in line:
                stats.pages_wired = int(line.split(":")[1].strip().rstrip("."))
            elif "Pages free" in line:
                stats.pages_free = int(line.split(":")[1].strip().rstrip("."))
            elif "page size of" in line:
                stats.page_size = int(line.split("page size of")[1].strip().split()[0])

        return stats
    except Exception:
        return PageCacheStats()


def print_cache_report(total_ram_gb: float = 64.0, model_overhead_gb: float = 10.0):
    """Print a human-readable page cache report."""
    stats = get_page_cache_stats()
    available = total_ram_gb - model_overhead_gb - 6.0  # 6GB OS overhead

    print("=== Page Cache Report ===")
    print(f"Active:   {stats.active_gb:.1f} GB")
    print(f"Inactive: {stats.inactive_gb:.1f} GB (≈ file cache)")
    print(f"Wired:    {(stats.pages_wired * stats.page_size / 1024**3):.1f} GB")
    print(f"Free:     {stats.free_gb:.1f} GB")
    print(f"Available for page cache: ~{available:.0f} GB")
    print(f"Expert cache capacity: ~{int(available * 1024 / 6.75)} experts")
    print("========================")
