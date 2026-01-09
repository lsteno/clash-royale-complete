# Focused Issues Summary

This document summarizes the specific issues we agreed to fix and why they
matter. It is intentionally narrow in scope.

## 1) Reset StateBuilder and RewardBuilder at Match End

What is wrong:
- The perception stack maintains internal history (HP baselines, unit tracking,
  OCR-derived state). If it is not reset between matches, old state bleeds into
  the next game and produces incorrect rewards and state features.

Why it matters:
- Reward drift and stale unit history cause the agent to learn from invalid
  signals. This is especially damaging for HP-based rewards, which rely on
  a clean baseline at match start.

Symptoms you see:
- HP deltas that appear before any combat.
- Rewards that do not correlate with visible tower damage.

Where to fix:
- `KataCR/katacr/policy/perceptron/reward_builder.py`
- `clash-royale-mbrl/src/perception/katacr_pipeline.py`

Plan:
- Ensure `StateBuilder.reset()` and `RewardBuilder.reset()` are called at every
  match end in both local and remote pipelines. The reset must happen before
  the next match begins.

## 2) Reduce Action-Effect Delay to Match Action FPS

What is wrong:
- Actions are applied at a lower cadence than frames are captured, so the
  observation stream can advance multiple frames before an action "lands."
  If the action cadence is 1/sec and FPS is 5, most frames are action-less.

Why it matters:
- The learner assumes each observation step corresponds to the most recent
  action, which is false. This weakens credit assignment and slows convergence.

Symptoms you see:
- Many "bad moves" that appear to do nothing.
- Learning stalls even after long training runs.

Where to address:
- `clash-royale-mbrl/scripts/remote_client_loop.py`
- `clash-royale-mbrl/src/environment/remote_bridge.py`

Plan:
- Make the effective action cadence explicit and align it with the training
  step rate. Reduce the delay so that each action has a chance to appear in
  the next observation used for learning.

## 3) Fix Tower-Destroy Tracking Bug (frame[i] vs frame[i, j])

What is wrong:
- `last_tower_destroy_frame` is a 2x2 grid indexed by [bel, side]. The current
  code uses `frame[i] = ...`, which updates the entire row (both towers on a
  side) instead of the specific tower cell.

Why it matters:
- This produces false tower-destroy detections and incorrect reward spikes,
  corrupting learning signals.

Where to fix:
- `KataCR/katacr/policy/perceptron/reward_builder.py`

Plan:
- Replace `frame[i] = self.frame_count` with `frame[i, j] = self.frame_count`.

## 4) Dynamic Target Masks for Spells

What is wrong:
- The action mask only checks card legality by elixir and slot, not whether a
  spell has a valid target. This lets the policy "waste" spells on empty tiles.

Why it matters:
- The agent spends elixir without effect, which both loses games and poisons
  the reward/action mapping it is trying to learn.

Where to fix:
- `clash-royale-mbrl/src/environment/action_mask.py`

Plan:
- Add target-aware masking: spells can only target enemy units/towers that are
  currently detected. If no valid target exists, all spell actions are masked.
