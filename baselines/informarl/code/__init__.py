"""InforMARL baseline — local agent–entity GNN + PPO (no CBF / no Lagrangian).

Uses the env GraphsTuple (agents + goals + lidar hits) already built by
DoubleIntegrator, with attention GNN aggregation (same family as GCBF actor
backbone) and PPO-clip. Safety only via soft cost in the training reward
(cost_weight); empirical only, not formal.
"""
