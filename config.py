from dataclasses import dataclass

@dataclass
class FLBOOptions:
    alpha: float = 10.0
    angle: float = 0.0
    tau: float = 0.5
    n_angles: int = 8

    learning_rate: float = 1e-3
    num_epochs: int = 100
    batch_size: int = 1