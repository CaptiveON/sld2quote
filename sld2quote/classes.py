"""Detector classes: one per symbol meaning, in a fixed order that the COCO export, the trainer and the trained reader all share.
The generator's device types map onto them; the reader maps a class back to a device type by its role on the sheet (a breaker above
the busbar is an ACB, one hanging from it an MCCB, one in the busbar a bus-section)."""

CLASSES = ["breaker", "fuse_switch", "isolator", "contactor", "transformer", "generator", "motor", "board",
           "ct", "meter", "relay", "spd", "capacitor", "earth"]

DEVICE_TO_CLASS = {"ACB": "breaker", "MCCB": "breaker", "BUS_SECTION": "breaker", "FUSE_SWITCH": "fuse_switch", "ISOLATOR": "isolator",
                   "CONTACTOR": "contactor", "TRANSFORMER": "transformer", "GENERATOR": "generator", "MOTOR": "motor", "BOARD": "board",
                   "CT": "ct", "METER": "meter", "RELAY": "relay", "SPD": "spd", "CAPACITOR": "capacitor", "EARTH": "earth"}

CLASS_ID = {name: i for i, name in enumerate(CLASSES)}  # category ids are 0-based everywhere in this repo
