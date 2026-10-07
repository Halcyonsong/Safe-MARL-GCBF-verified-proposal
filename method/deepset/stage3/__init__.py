"""Stage3: counterexample-guided repair of the Stage1 DeepSet policy.

Stage2 owns formal verification (L1/L2 queries). Stage3 owns CE collection and
feedback training. Domain / dynamics helpers remain in ``stage2`` and are
imported here so verify and repair share one physics contract.
"""
