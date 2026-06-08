# M4 anchor: EnsembleSelector — s_ens = α·minmax(cos) + (1-α)·minmax(σ(GAT)), α=0.5, top_k=20.
from .ensemble_selector import EnsembleSelector

__all__ = ["EnsembleSelector"]
