"""Model downloading utilities."""

import urllib.request
import shutil


def download_model(url, dest_path, description, min_size=1_000_000):
    """Download a model file if it doesn't already exist."""
    from app.core.logging import UnifiedLogger
    from app.core.utils import safe_execute_with_retry

    logger = UnifiedLogger()

    if dest_path.is_file() and dest_path.stat().st_size >= min_size:
        return

    def download_func():
        logger.info(f"Downloading {description}",
                    extra={'url': url, 'dest': dest_path})
        req = urllib.request.Request(url,
                                    headers={"User-Agent": "Mozilla/5.0"})
        with (urllib.request.urlopen(req, timeout=60) as resp,
              open(dest_path, "wb") as out):
            shutil.copyfileobj(resp, out)
        if not dest_path.exists() or dest_path.stat().st_size < min_size:
            raise RuntimeError(f"Downloaded {description} seems incomplete")
        logger.success(f"{description} downloaded successfully.")

    try:
        safe_execute_with_retry(download_func)
    except Exception as e:
        logger.error(f"Failed to download {description}", exc_info=True,
                     extra={'url': url})
        raise RuntimeError(
            f"Failed to download required model: {description}"
        ) from e
