"""Where the repo root is — ONE definition.

The package lives one level below the checkout; config.json, static/,
ytdlp_plugins/, ffmpeg/bin and the Windows data/ + models/ defaults all live
AT the checkout. Every module that used to say dirname(__file__) (which
meant the repo root while the modules were flat) reads REPO_ROOT instead.

os.path.abspath, not Path.resolve(): the flat layout never followed symlinks
and a symlinked checkout must keep resolving the same paths.
"""
import os

REPO_ROOT: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
