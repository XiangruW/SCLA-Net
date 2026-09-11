"""SCLA-Net: Structure-aware Context and Localization Aggregation Network.

Assembles the lightweight depthwise-separable encoder-decoder backbone with
SCEA, SALT and LGCA as described in Sections "Overall Architecture",
"Lightweight Backbone Network", "SCEA", "SALT" and "LGCA" of the manuscript.

The class also exposes the four ablation variants reported in the
component-wise ablation study through the ``use_scea / use_salt / use_lgca``
switches:

===========================  ==========================================
configuration                behaviour
===========================  ==========================================
baseline                     none of the three modules
+SCEA                        SCEA only
+SALT                        SALT only (tokens are not consumed downstream)
+LGCA                        LGCA with ``K`` learnable query tokens
+ALL                         SCEA + SALT + LGCA
===========================  ==========================================
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import ConvNormAct3d, DepthwiseSeparableConv3d, ResidualDSBlock
from .lgca import LGCA
from .salt import SALT
from .scea import SCEA


class SCLANet(nn.Module):
    """Structure-aware Context and Localization Aggregation Network.

    Parameters
    ----------
    in_channels:
        Number of input image channels (1 for CT/MRI).
    num_classes:
        Number of output classes, including the background.
    encoder_channels:
        Channel width of the four encoder stages.
    decoder_channels:
        Channel width of the four decoder stages.
    bottleneck_channels:
        Channel width of the bottleneck (512 in the manuscript).
    blocks_per_stage:
        Number of depthwise-separable residual blocks per stage.
    num_tokens / token_dim:
        Number of localization tokens ``K`` and their dimensionality.
    scea_reduction / scea_bottleneck_reduction:
        Reduction ratios of SCEA in the encoder-decoder stages and at the
        bottleneck (16 and 4 in the manuscript).
    salt_kwargs:
        Extra keyword arguments forwarded to :class:`~sclanet.models.salt.SALT`.
    use_scea / use_salt / use_lgca:
        Ablation switches (see the class docstring).
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 14,
        encoder_channels: Sequence[int] = (48, 96, 192, 384),
        decoder_channels: Sequence[int] = (256, 128, 64, 48),
        bottleneck_channels: int = 512,
        blocks_per_stage: int = 2,
        num_tokens: int = 16,
        token_dim: int = 256,
        scea_reduction: int = 16,
        scea_bottleneck_reduction: int = 4,
        scea_attn_grid: int = 8,
        salt_kwargs: Dict | None = None,
        use_scea: bool = True,
        use_salt: bool = True,
        use_lgca: bool = True,
    ) -> None:
        super().__init__()
        self.use_scea = use_scea
        self.use_salt = use_salt
        self.use_lgca = use_lgca
        self.num_tokens = num_tokens
        self.token_dim = token_dim

        enc = list(encoder_channels)
        dec = list(decoder_channels)

        # ----------------------------- encoder ---------------------------- #
        self.stem = DepthwiseSeparableConv3d(
            in_channels, enc[0], stride=2, kernel_size=3
        )  # full resolution -> /2

        self.enc_stages = nn.ModuleList()
        self.enc_down = nn.ModuleList()
        self.enc_scea = nn.ModuleList()

        for i, out_ch in enumerate(enc):
            # the input of stage i has enc[i] channels (produced by the stem for
            # the first stage and by the preceding downsampling block otherwise)
            blocks: List[nn.Module] = [
                ResidualDSBlock(out_ch, out_ch, stride=1) for _ in range(blocks_per_stage)
            ]
            self.enc_stages.append(nn.Sequential(*blocks))
            self.enc_scea.append(
                SCEA(out_ch, reduction=scea_reduction, attn_grid=scea_attn_grid)
                if use_scea
                else nn.Identity()
            )
            if i < len(enc) - 1:
                self.enc_down.append(DepthwiseSeparableConv3d(out_ch, enc[i + 1], stride=2))

        # ----------------------------- bottleneck ------------------------- #
        self.bottleneck = nn.Sequential(
            ConvNormAct3d(enc[-1], bottleneck_channels, kernel_size=1, norm="layer", act="leakyrelu"),
            DepthwiseSeparableConv3d(bottleneck_channels, bottleneck_channels),
        )
        self.bottleneck_scea = (
            SCEA(
                bottleneck_channels,
                reduction=scea_bottleneck_reduction,
                attn_grid=scea_attn_grid,
            )
            if use_scea
            else nn.Identity()
        )

        # --------------------------- SALT and LGCA ------------------------ #
        salt_kwargs = dict(salt_kwargs or {})
        self.salt = (
            SALT(bottleneck_channels, num_tokens=num_tokens, token_dim=token_dim, **salt_kwargs)
            if use_salt
            else None
        )
        self.learnable_tokens = (
            nn.Parameter(torch.randn(1, num_tokens, token_dim) * 0.02)
            if (use_lgca and not use_salt)
            else None
        )
        self.lgca = (
            LGCA(bottleneck_channels, token_dim=token_dim, out_channels=bottleneck_channels)
            if use_lgca
            else None
        )

        # ----------------------------- decoder ---------------------------- #
        self.dec_stages = nn.ModuleList()
        self.dec_scea = nn.ModuleList()
        enc_skips = [enc[-1], enc[-2], enc[-3], enc[0]]
        in_ch = bottleneck_channels
        for i, out_ch in enumerate(dec):
            skip_ch = enc_skips[i]
            blocks = [
                ResidualDSBlock(in_ch + skip_ch, out_ch, stride=1),
                ResidualDSBlock(out_ch, out_ch, stride=1),
            ]
            self.dec_stages.append(nn.Sequential(*blocks))
            self.dec_scea.append(
                SCEA(out_ch, reduction=scea_reduction, attn_grid=scea_attn_grid)
                if use_scea
                else nn.Identity()
            )
            in_ch = out_ch

        self.head = nn.Conv3d(dec[-1], num_classes, kernel_size=1, bias=True)

        self._init_weights()

    # ------------------------------------------------------------------ #
    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="leaky_relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        if self.head.bias is not None:
            nn.init.zeros_(self.head.bias)

    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Returns a dictionary with the segmentation logits and, when the
        corresponding modules are enabled, the anatomical response map, the
        selected centers, the aggregation weights and the LGCA attention.
        """
        skips: List[torch.Tensor] = []
        input_size = x.shape[2:]

        x = self.stem(x)  # /2
        skips.append(x)

        for i, stage in enumerate(self.enc_stages):
            x = stage(x)
            x = self.enc_scea[i](x)
            if i < len(self.enc_down):
                x = self.enc_down[i](x)
                skips.append(x)

        # ------------------------------ deepest ---------------------------- #
        x = self.bottleneck(x)
        x = self.bottleneck_scea(x)

        out: Dict[str, torch.Tensor] = {}
        tokens = None
        if self.salt is not None:
            tokens, response_logits, centers, weights = self.salt(x)
            out["response_logits"] = response_logits
            out["response"] = torch.sigmoid(response_logits)
            out["centers"] = centers
            out["weights"] = weights
        if self.lgca is not None:
            if tokens is None:  # LGCA without SALT -> learnable queries
                tokens = self.learnable_tokens.expand(x.shape[0], -1, -1)
            x, attn = self.lgca(tokens, x)
            out["attn"] = attn
            out["tokens"] = tokens

        # ------------------------------ decoder ---------------------------- #
        for i, stage in enumerate(self.dec_stages):
            skip = skips[-1 - i]
            x = F.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
            x = stage(x)
            x = self.dec_scea[i](x)

        if x.shape[2:] != input_size:  # restore the input resolution
            x = F.interpolate(x, size=input_size, mode="trilinear", align_corners=False)

        out["logits"] = self.head(x)
        return out


def build_model(cfg: Dict, variant: str | None = None) -> SCLANet:
    """Build a model from a configuration dictionary.

    ``variant`` selects one of the ablation settings reported in the
    manuscript: ``baseline``, ``scea``, ``salt``, ``lgca`` or ``all``.
    """
    model_cfg = dict(cfg.get("model", {}))
    model_cfg.pop("variant", None)

    if variant is None:
        variant = cfg.get("variant", "all")
    variant = (variant or "all").lower()

    flags = {
        "baseline": (False, False, False),
        "scea": (True, False, False),
        "salt": (False, True, False),
        "lgca": (False, False, True),
        "all": (True, True, True),
    }
    if variant not in flags:
        raise ValueError(f"unknown variant '{variant}', expected one of {sorted(flags)}")
    use_scea, use_salt, use_lgca = flags[variant]

    return SCLANet(
        use_scea=use_scea,
        use_salt=use_salt,
        use_lgca=use_lgca,
        **model_cfg,
    )
