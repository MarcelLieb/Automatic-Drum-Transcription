from collections import defaultdict
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
    ("SD",),  # Snare Drum
    ("CHH", "PHH", "OHH"),  # Hi-Hat
)
three_class_names = ("BD", "SD", "HH")

four_class_mapping = (
    ("BD",),  # Bass Drum
    ("SD",),  # Snare Drum
    ("CHH", "PHH", "OHH"),  # Hi-Hat
    ("LT", "MT", "HT"),  # Toms
)
four_class_names = ("BD", "SD", "HH", "TT")

# Mapping used in ADTOF
five_class_mapping = (
    ("BD",),  # Bass Drum
    ("SD",),  # Snare Drum
    ("CHH", "PHH", "OHH"),  # Hi-Hat
    ("LT", "MT", "HT"),  # Toms
    ("CRC", "SPC", "CHC", "RD", "RB"),  # Cymbal + Ride
)
five_class_names = ("BD", "SD", "HH", "TT", "CY + RD")

eight_class_mapping = (
    ("BD",),  # Bass Drum
    ("SD",),  # Snare Drum
    ("LT", "MT", "HT"),  # Toms
    ("CHH", "PHH", "OHH"),  # Hi-Hat
    ("CRC", "SPC", "CHC"),  # Cymbal
    ("RD",),  # Ride
    ("RB", "CB"),  # Bell
    ("CL",),  # Clave/Sticks
)
eight_class_names = ("BD", "SD", "TT", "HH", "RD", "BE", "CY", "CL")

eighteen_class_mapping = tuple([(key,) for key in drum_midi_mapping.keys()])
eighteen_class_names = tuple(drum_midi_mapping.keys())


class DrumMapping(Enum):
    THREE_CLASS = (three_class_mapping, three_class_names)
    THREE_CLASS_STANDARD = (three_class_standard_mapping, three_class_names)
    FOUR_CLASS = (four_class_mapping, four_class_names)
    FIVE_CLASS = (five_class_mapping, five_class_names)
    EIGHT_CLASS = (eight_class_mapping, eight_class_names)
    EIGHTEEN_CLASS = (eighteen_class_mapping, eighteen_class_names)

    def __len__(self):
        return len(self.value[0])

    def __getitem__(self, item):
        return self.value[item][0]

    def get_name(self, item):
        return self.value[item][1]

    def __str__(self):
        return self.name.lower().replace("_", " ").capitalize()

    def __repr__(self):
        return self.__str__()

    @staticmethod
    def prettify(class_desc: tuple[str, ...]):
        return " + ".join(class_desc)

    @staticmethod
    def from_str(desc: str):
        for mapping in DrumMapping:
            mapping = mapping[0]
            if str(mapping).lower().replace("_", " ").capitalize() == desc.lower().replace("_", " ").capitalize():
                return mapping
        raise ValueError(f"Invalid mapping description: {desc}")

    def get_midi_to_class(self):
        reverse_map = np.zeros(128, dtype=int)
        reverse_map.fill(-1)
        for idx, drum_classes in enumerate(self.value[0]):
            for drum_class in drum_classes:
                for pitch in drum_midi_mapping[drum_class]:
                    reverse_map[pitch] = idx
        return reverse_map

    def get_name_to_class_number(self):
        dic = defaultdict(lambda: -1)
        for idx, drum_classes in enumerate(self.value[0]):
            for drum_class in drum_classes:
                dic[drum_class] = idx
        for key in drum_midi_mapping.keys():
            if key not in dic.keys():
                dic[key] = -1
        return dic
