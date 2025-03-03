from importlib.metadata import version

__version__ = version("nideconv")

from .response_fitter import ResponseFitter
from .group_analysis import GroupResponseFitter
