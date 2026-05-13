
### Experiment

Policies like AccDiff / ClassDist maximize Oracle gain, however MWM policies maximize non-Oracle gain.

AccDiff/ClassDist:
  - oracle gain keeps increasing strongly
  - non-oracle gain stays weak

MWM_AccDiff:
  - oracle gain eventually becomes negative
  - but non-oracle gain becomes extremely strong


## A) Patience-Based Oracle Blocking

To reduce negative Oracle gain during later training rounds, we introduced a patience-based Oracle blocking mechanism. After each Oracle-supervised update, the student’s validation accuracy gain is measured:
Δoracle = Acc(after) − Acc(before). If the Oracle gain is negative, a counter for that model is increased; otherwise, the counter is reset. Once a model reaches a predefined patience threshold (patience = 2), it is blocked from future Oracle pairings while still participating in peer-to-peer KD training.


