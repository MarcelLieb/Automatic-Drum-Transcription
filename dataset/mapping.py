from enum import Enum

import numpy as np

# Mapping identical to the one used in "Towards multi-instrument drum transcription"
drum_midi_mapping = {
    "BD": [35, 36],
    "SD": [38, 40],
    "SS": [37],
    "CLP": [39],
    "LT": [41, 43],
    "MT": [45, 47],
    "HT": [48, 50],
    "CHH": [42],
    "PHH": [44],
    "OHH": [46],
    "TB": [54],
    "RD": [51, 59],
    "RB": [53],
    "CB": [56],
    "CRC": [49, 57],
    "SPC": [55],
    "CHC": [52],
    "CL": [75],
}

drum_midi_mapping = {key: tuple(value) for key, value in drum_midi_mapping.items()}

three_class_mapping = (
    ("BD",),  # Bass Drum
    ("SD", "SS", "CLP"),  # Snare Drum + alike
    ("CHH", "PHH", "OHH"),  # Hi-Hat
)

# Commonly used mapping
three_class_standard_mapping = (
    ("BD",),  # Bass Drum
    ("SD",),  # Snare Drum + alike
    ("CHH", "PHH", "OHH"),  # Hi-Hat
)

four_class_mapping = (
    ("BD",),  # Bass Drum
    ("SD", "SS", "CLP"),  # Snare Drum + alike
    ("CHH", "PHH", "OHH"),  # Hi-Hat
    ("LT", "MT", "HT"),  # Toms
)

# Mapping used in ADTOF
five_class_mapping = (
    ("BD",),  # Bass Drum
    ("SD", "SS", "CLP"),  # Snare Drum + alike
    ("CHH", "PHH", "OHH"),  # Hi-Hat
    ("LT", "MT", "HT"),  # Toms
    ("CRC", "SPC", "CHC", "RD"),  # Cymbal + Ride
)

eight_class_mapping = (
    ("BD",),  # Bass Drum
    ("SD", "SS", "CLP"),  # Snare Drum + alike
    ("LT", "MT", "HT"),  # Toms
    ("CHH", "PHH", "OHH"),  # Hi-Hat
    ("CRC", "SPC", "CHC"),  # Cymbal
    ("RD",),  # Ride
    ("RB", "CB"),  # Bell
    ("CL",),  # Clave/Sticks
)

eighteen_class_mapping = tuple([(key,) for key in drum_midi_mapping.keys()])


class DrumMapping(Enum):
    THREE_CLASS = three_class_mapping
    THREE_CLASS_STANDARD = three_class_standard_mapping
    FOUR_CLASS = four_class_mapping
    FIVE_CLASS = five_class_mapping
    EIGHT_CLASS = eight_class_mapping
    EIGHTEEN_CLASS = eighteen_class_mapping

    def __len__(self):
        return len(self.value)


def get_midi_to_class(mapping: tuple[tuple[str, ...], ...]):
    reverse_map = np.zeros(128)
    reverse_map.fill(-1)
    for idx, drum_classes in enumerate(mapping):
        for drum_class in drum_classes:
            for pitch in drum_midi_mapping[drum_class]:
                reverse_map[pitch] = idx
    return reverse_map


def get_name_to_class_number(mapping: DrumMapping):
    dic = {}
    for idx, drum_classes in enumerate(mapping.value):
        for drum_class in drum_classes:
            dic[drum_class] = idx
    for key in drum_midi_mapping.keys():
        if key not in dic.keys():
            dic[key] = -1
    return dic
