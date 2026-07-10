# EAL Surtur

Epoch's spatial training and capability-preservation runner.

## Overview
Named after the Norse giant of fire, **Surtur** is the post-training runner designed to constrain the "fire" of parameter optimization (alignment, RL/GRPO) to the behavioral periphery (layers 24–27), keeping the core capabilities of the model frozen and insulated.

## Rationale
- **Why Surtur:** Post-training SFT or GRPO typically backpropagates gradients across the entire model, degrading the central logic engine. Surtur acts as the tempering shield, confining updates to target layers.
- **Goal:** Enable compute-efficient, capability-preserving alignment on a single GPU.
