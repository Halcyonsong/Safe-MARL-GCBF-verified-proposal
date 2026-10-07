# Official GCBF+ Baseline (comparison)

Official MIT-REALM GCBF+ GNN Actor + candidate CBF from:

```text
baselines/gcbfplus/log/gcbf+/seed0_20261001232611/models/1000/
```

This is the **comparison baseline / teacher**, not the method Stage1.
It provides rollout actions used to distill the Stage1 DeepSet student.

Independent empirical eval (32 episodes):

```text
safe ≈ 95.7%, finish ≈ 95.3%, success ≈ 91.4%
```

Keep for later baseline comparisons. Not formally verified.
