from controller import Controller
import numpy as np


def normalize_inputs(inputs):
    min_v = min(inputs)
    max_v = max(inputs)

    scale_factor = (max_v - min_v)

    # don't normalize the boolean sensor inputs, index 2 and 3, which is the direction they face
    indexes_to_normalize = [0, 1] + [i for i in range(4, 20)]
    for i in indexes_to_normalize:
        inputs[i] = (inputs[i] - min_v) / scale_factor

    return inputs


class PlayerController(Controller):
    def __init__(self):
        pass

    def control(self, inputs, ffn):
        # prepare input
        inputs = normalize_inputs(inputs)

        output = ffn.activate(inputs)

        # return output
        return np.round(output).astype(int)
