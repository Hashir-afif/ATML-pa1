"""ERM (Task 3) = the Task 2 Source-only model. It is LOADED from its Task 2 checkpoint and never retrained.

The objective is L_ERM = (1/3) sum_e R_e(theta) with domain-balanced batches (8 per source domain), which is
exactly Task 2's Source-only loss; the class is re-exported so the definition lives in one place.
"""
from task2.methods.source_only import SourceOnly as ERM  # noqa: F401

ERM_CHECKPOINT_DIR = "task2/checkpoints/source_only"
