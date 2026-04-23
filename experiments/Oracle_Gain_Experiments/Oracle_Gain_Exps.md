Oracle Gain Diagnostic Experiments

To investigate the source of the observed negative oracle gain, we design a set of controlled ablation experiments. Each experiment isolates a single factor in the training dynamics to determine whether the issue arises from optimization behavior, learning rate configuration, or interactions between learning modes.

A. Reset Optimizer State for Oracle Updates.
In the default setting, each model retains its optimizer state across rounds, including accumulated momentum and gradient history. However, since models alternate between oracle-guided (supervised) and peer-to-peer (distillation) updates, this carryover may introduce instability. To test this, we reset the optimizer entirely whenever a model is paired with the oracle, ensuring that the oracle-guided update starts from a fresh optimization state. Improvement in oracle gain under this condition would indicate that optimizer state carryover is a contributing factor.

B. Reduce Learning Rate for Oracle-Guided Updates.
Oracle-guided updates use ground-truth labels and may produce large gradient steps, especially under a relatively high learning rate. This can lead to overshooting and reduced generalization performance. To evaluate this, we reduce the learning rate only for oracle-guided updates (e.g., to 0.01 or 0.001), while keeping peer-to-peer learning unchanged. If oracle gain stabilizes, it suggests that excessive step sizes are responsible for the observed degradation.

C. Remove Momentum for Oracle Updates.
Momentum accumulates gradients from previous updates, which may not align with the objective of oracle-guided learning. To isolate its effect, we use stochastic gradient descent without momentum specifically for oracle updates, while keeping momentum unchanged for peer learning. If this modification improves oracle gain, it implies that stale momentum is interfering with effective supervised updates.

D. Measure Training and Validation Behavior Around Oracle Steps.
To distinguish between optimization issues and generalization problems, we track both training loss and validation accuracy before and after each oracle-guided update. If training loss decreases while validation accuracy drops, the model is fitting the training data but generalizing worse, indicating overshooting or overfitting. This helps clarify whether negative oracle gain reflects a failure to learn or a degradation in generalization.

E. Remove Learning Rate Scheduler.
The default training setup uses a learning rate scheduler that decays the learning rate at predefined milestones. However, because models are updated intermittently rather than continuously, the scheduler may not align well with the effective training dynamics. To test this, we remove the scheduler entirely. If oracle gain behavior changes, it suggests that the scheduling strategy contributes to instability.

F. Remove KL Divergence from Peer-to-Peer Learning.
In the standard setup, peer-to-peer learning combines cross-entropy with a KL divergence (distillation) term. These objectives may introduce conflicting gradient signals across rounds, particularly when followed by oracle-guided updates. To isolate this effect, we remove the KL divergence component and retain only the supervised loss during peer interactions. Improvement in oracle gain would indicate that the distillation objective interferes with subsequent oracle learning.
