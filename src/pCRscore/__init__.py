"""
Predicts pathological Complete Response scores

To use this software, load one of the available modules listed under
"PACKAGE CONTENTS".
"""

__author__ = "Youness Azimzade, Waldir Leoncio Netto"

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
