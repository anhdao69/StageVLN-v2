# v2/v3 acceptance record

Validated on 2026-09-18 inside the existing four H100 interactive allocation.
The [implementation report](v2_v3_implementation_report.md) describes each test's
scope, numerical tolerance, production settings, and measured timing.

| Category | Status | Evidence and limit |
|---|---|---|
| Architecture and causal data flow | Passed tested contracts | Native adapter parity, M64 external writer, R0/R4 selection, masked memory, observation-only inputs, exact frame identities |
| Temporal gradients and boundaries | Passed tested contracts | K8 tiny actual-Qwen direct/bridge gradient comparison, delayed-write gradients, detach/reset tests; full-model K2 selected-gradient equality |
| Distributed objective and checkpointing | Passed tested contracts | Two-rank union objective including idle ranks/tails; exact full-data schedule; tiny-model resume; actual four-GPU checkpoint continuation |
| Tiny overfit | Passed | Delayed-cue loss 1.4193 to 0.0018; complete real-episode tiny-model loss 12.4074 to 6.1005 after 18 passes |
| Full-model execution and memory | Passed bounded runs | Four-H100 v2/v3 training and long-instruction stress; settings and exact peaks in report |
| Export, actor, and diagnostic trace | Passed tested contracts | Exact reload of 833 saved tensors, tied weights, valid/idempotent actions, controlled memory interventions, fixed-policy 23-observation trace |
| Full epoch and matched-control quality | Not run | Prepared R2R configs; smoke runs do not establish convergence or comparative quality |
| Development closed-loop quality | Not run | No simulator SR/SPL/NE/OS or unseen-environment result |
| End-to-end Hugging Face publication | Not run | Completed-training upload guard and private-repository helper prepared |

The implementation is ready for a monitored full-training run with the recorded
settings. This is not a claim of universal correctness or navigation improvement.
No new Slurm submission or full-training run was performed.
