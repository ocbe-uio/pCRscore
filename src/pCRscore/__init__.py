# Fetch version number from pyproject.toml

try:
    # Try to fetch the version for installed packages
    from importlib.metadata import version
    __version__ = version("pCRscore")
except ImportError:
    # Fallback for Python<3.8 where importlib.metadata might not be available
    try:
        from importlib_metadata import version
        __version__ = version("pCRscore")
    except ImportError:
        # Set a default version or leave it undefined if you prefer
        __version__ = "unknown"


# Load modules
from . import discovery_svm
