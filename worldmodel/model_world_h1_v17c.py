
# -*- coding: utf-8 -*-
"""
MagWorld v17c — v17 MagFluid with the fluid branch switched OFF at init.

v17 (fl_mix init 0 -> sigmoid 0.5) collapsed raw cosine to 0.20 vs v13's
0.39: starting with transport at 50% pushes the model off v13's sharp-response
basin during co-adaptation, and the training loss (DE-focused) never punishes
the resulting diffuseness.  v17c starts the fluid at sigmoid(-4) ~ 0.018 so
the model begins life as (almost exactly) v13 and can open the transport
valve only if the data actually rewards it.  Learned fl_mix after training is
then a direct readout of whether fluid transport carries signal.
"""

from __future__ import annotations

import torch

from model_world_h1_v17 import WorldModelH1V17


class WorldModelH1V17C(WorldModelH1V17):
    """v17 with fluid branch closed at initialization."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        with torch.no_grad():
            self.fl_mix.fill_(-4.0)      # sigmoid ~ 0.018: start as v13
            self.fl_dt_raw.fill_(-2.0)   # dt ~ 0.12 per step when it opens


WorldModel = WorldModelH1V17C
