"""Small, dependency-free artifact downloader."""

from pathlib import Path
import shutil
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import urlopen


def download_file(
    url: str,
    out_dir: str | Path = ".",
    backup_url: str | None = None,
) -> Path:
    """Download a file, falling back to a backup URL on transport errors."""
    destination_dir = Path(out_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)
    filename = Path(urlparse(url).path).name
    destination = destination_dir / filename

    errors: list[tuple[str, Exception]] = []
    for candidate in filter(None, (url, backup_url)):
        try:
            with urlopen(candidate, timeout=30) as response:
                remote_size = int(response.headers.get("Content-Length", 0))
                if (
                    destination.exists()
                    and remote_size
                    and destination.stat().st_size == remote_size
                ):
                    return destination
                temporary = destination.with_suffix(destination.suffix + ".part")
                with temporary.open("wb") as output:
                    shutil.copyfileobj(response, output, length=1024 * 1024)
                temporary.replace(destination)
                return destination
        except (OSError, URLError) as error:
            errors.append((candidate, error))

    details = "\n".join(
        f"- {candidate}: {type(error).__name__}: {error}" for candidate, error in errors
    )
    raise RuntimeError(f"Failed to download {filename}:\n{details}") from errors[-1][1]
