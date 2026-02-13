"""

CNNAMON is a modular Python framework for building, training, and interpreting\
1D Convolutional Neural Networks (CNNs) for DNA sequence analysis.
The framework integrates data preparation, model construction, and rich \
interpretability tools in a unified and flexible system tailored for genomics research.

"""

from . import CNN1D
from . import utility

__all__ = ["CNN1D", "utility"]
__version__ = "0.2.2"
