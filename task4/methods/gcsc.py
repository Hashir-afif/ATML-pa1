"""GCSC: exactly the Vanilla recipe with one change, RandAugment(num_ops=2, magnitude=9) after crop and flip.

Initialisation, optimiser, schedule, batch size, epochs, seed and checkpoint rule are unchanged.
"""
from task4.methods.vanilla import Vanilla


class GCSC(Vanilla):
    name = "gcsc"
    use_randaugment = True
