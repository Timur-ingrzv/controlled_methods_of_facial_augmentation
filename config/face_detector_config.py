from dataclasses import dataclass

@dataclass
class DetectorConfig:
    box_expand_left = 40
    box_expand_right = 40
    box_expand_top = 100
    box_expand_bottom = 0

    margin = 30