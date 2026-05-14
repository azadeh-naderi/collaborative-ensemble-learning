
### Experiment

Policies like AccDiff / ClassDist maximize Oracle gain, however MWM policies maximize non-Oracle gain.

AccDiff/ClassDist:
  - oracle gain keeps increasing strongly
  - non-oracle gain stays weak

MWM_AccDiff:
  - oracle gain eventually becomes negative
  - but non-oracle gain becomes extremely strong

from plot, this looks like a compensation trade-off:

Final gain≈Oracle gain+Non-oracle gain

So when OG goes up, NG goes down; when NG goes up, OG goes down. That is why final accuracy can stay almost the same.

The key point is: you may not be able to maximize both independently with one fixed policy, because the two gains compete for the same limited resources:

  - only one update per student per round
  - only one oracle interaction per round
  - same models are used for both oracle and peer learning
  - oracle training can make models more similar, reducing useful peer diversity
  - peer diffusion can move models away from oracle-aligned direction, reducing later oracle benefit

So the goal should not be simply:
  - maximize OG and maximize NG separately

but rather:
  - maximize non-overlapping / complementary OG and NG.

That means you want oracle learning to improve things that peer learning cannot, and peer learning to improve things that oracle learning does not already fix.

# Better Research Question

Instead of asking: Can I get maximum OG and maximum NG?

ask:Can I reduce redundancy between oracle learning and peer learning so both contribute unique improvements?

That is probably the real issue.


## A) Patience-Based Oracle Blocking

To reduce negative Oracle gain in MWM pairing policies, we used a patience-based Oracle blocking mechanism. After each Oracle update, the student’s validation accuracy gain is measured:
Δoracle = Acc(after) − Acc(before). If the Oracle gain is negative, a counter for that model is increased; otherwise, the counter is reset. Once a model reaches a predefined patience threshold (patience = 2), that model is blocked from future Oracle pairings while still participating in peer-to-peer KD training.

## B) Adaptive Switching via Average Oracle Gain

This experiment introduces an adaptive pairing strategy that dynamically switches policies when Oracle learning becomes unstable during collaborative ensemble training.

Training begins with an Oracle-focused policy such as AccDiff, which typically produces strong early learning gains by pairing weak learners with stronger models or the Oracle. During training, the framework continuously tracks Oracle gain: oracle_gain = student_accuracy_after_oracle - student_accuracy_before_oracle

A moving average of recent Oracle gains is computed over a sliding window. If the moving-average Oracle gain remains negative for several consecutive rounds (patience), the training policy automatically switches to another strategy such as MWM_AccDiff.

Example configuration:
  - pairing_strategy = "AccDiff"
  - switch_to = "MWM_AccDiff"
  - window_size = 5
  - oracle_negative_patience = 3
  - oracle_negative_threshold = 0.0
  - min_round_before_switch = 40

The motivation is that:

AccDiff often maximizes early Oracle learning, while MWM_AccDiff tends to improve later-stage peer-to-peer learning and knowledge diffusion.

Using a moving average helps avoid switching due to noisy single-round fluctuations, while patience ensures switching only occurs when negative Oracle behavior persists consistently.



