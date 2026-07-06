from enum import Enum

class State(Enum):
    IDLE = 0
    PLANNING = 1
    SCORING = 2
    NAVIGATING = 3