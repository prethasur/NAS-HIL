"""TINAS-ShipDet: Tiny Neural Architectures for Onboard SAR Ship Detection."""
__version__ = "1.0.0"

# Fix OpenMP duplicate library crash on Windows (Anaconda + PyTorch)
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
