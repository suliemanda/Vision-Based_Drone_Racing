import os

import yaml

from .gate import Gate
class Track:
    def __init__(self, track_layout_yaml):
        print("Loading track layout from [%s]" % track_layout_yaml)

        self.gates = []

        assert os.path.isfile(track_layout_yaml), track_layout_yaml
        with open(track_layout_yaml, "r") as stream:
            track_yaml = yaml.safe_load(stream)
            for i in range(track_yaml["gates"]["N"]):
                gate_str = "Gate" + str(i + 1)
                curr_gate = Gate(
                    track_yaml["gates"][gate_str]["position"],
                    track_yaml["gates"][gate_str]["rotation"],
                )
                self.gates.append(curr_gate)

    def draw(self, ax):
        for gate in self.gates:
            gate.draw(ax)
