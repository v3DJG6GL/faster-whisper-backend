"""Where the repo root is — ONE definition.

The package lives one level below the checkout; config.json, static/,
ytdlp_plugins/, ffmpeg/bin and the Windows data/ + models/ defaults all live
AT the checkout; modules read REPO_ROOT instead of their own dirname(__file__).

os.path.abspath, not Path.resolve(): symlinks are deliberately not followed,
so a symlinked checkout keeps resolving the same paths.
"""
import os

REPO_ROOT: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
