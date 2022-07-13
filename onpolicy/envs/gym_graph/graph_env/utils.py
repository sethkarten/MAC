import dataclasses
import enum

import numpy as np


class MaxEnum(enum.IntEnum):
    @classmethod
    def get_max(cls):
        return max(x.value for x in cls)


class OneHotEnum(MaxEnum):

    def to_one_hot(self):
        raise NotImplementedError()

    @classmethod
    def one_hot_dim(cls) -> int:
        raise NotImplementedError()


class Role(OneHotEnum):
    ENGINEER = 0
    MEDIC = 1
    TRANSPORTER = 2

    @classmethod
    def one_hot_dim(cls):
        return 3

    def to_one_hot(self):
        res = np.zeros(3)
        res[self.value] = 1
        return res


class Actions(MaxEnum):
    CLEAN = 0
    STABILIZE = 1
    PICK = 2
    EVACUATE = 3
    STILL = 4

    @classmethod
    def from_RL(cls, num_nodes, action: int):
        return cls(action - num_nodes)

    def to_RL(self, num_nodes):
        return self.value + num_nodes


class InjuryIndex(OneHotEnum):
    UNKNOWN = 0
    A = 1
    B = 2
    C = 3
    REGULAR = 4

    @classmethod
    def one_hot_dim(cls):
        return 3

    def to_one_hot(self):
        res = np.zeros(3)
        res[self.value - 1] = 1
        return res


@dataclasses.dataclass(frozen=True)
class World:
    """
    Constants
    """

    BASE_SPEED: float = 4.317  # Base speed of minecraft

    # Size in pixels of a tile in the full-scale human view
    TILE_PIXELS: int = 32

    # How far away can players stay from a yellow victim?
    HIGH_PRIORITY_RADIUS: int = 1

    # Makes four decisions within a second.
    # This is consistent with previous artificial players in the lab
    SECONDS_TO_STEPS: int = 4

    RUBBLE_STEPS: int = 4  # remember to make this consistent with SECONDS_TO_STEPS for Saturn!

    VICTIM_STEPS: int = 12  # remember to make this consistent with SECONDS_TO_STEPS for Saturn!

    ALL_ROLE_CAN_TRANSPORT: bool = True

    RUBBLE_BLOCK_VIEW: bool = False

    FULLY_OBSERVABLE: bool = False
