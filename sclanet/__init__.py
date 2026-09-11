"""SCLA-Net: Structure-aware Context and Localization Aggregation Network.

Reference implementation for the manuscript

    SCLA-Net: A Lightweight Structure-aware Context and Localization
    Aggregation Network for 3D Medical Image Segmentation

Modules
-------
models      network, SCEA, SALT, LGCA and the lightweight backbone
data        preprocessing, datasets and data splits
losses      Dice + cross-entropy + localization-aware auxiliary loss
metrics     Dice similarity coefficient and 95% Hausdorff distance
inference   sliding-window inference
utils       config handling, seeding, logging, checkpoint helpers
"""

__version__ = "1.0.0"

__all__ = ["models", "data", "losses", "metrics", "inference", "utils"]
