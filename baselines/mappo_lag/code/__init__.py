"""MAPPO-Lagrangian baseline (CMDP soft constraint; empirical only).

Homogeneous agents, parameter sharing, PPO-clip + dual λ on env cost.
Observation: same 44-D mean-field features as MF track (fair task featurization).
Not formal verification / not CBF-QP.
"""
