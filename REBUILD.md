# Provenance and rebuild log

Source: sdm0p/tredence-aqua, commit 6999a8605b85cbb6434936d291cf24f4465a2e59.

This is a transparent staged migration/rebuild with additional verification. Source from the first implementation is reused. Commits use actual current timestamps. The user deleted and recreated the GitHub repository; earlier history is retained only in local repositories/backup branches, not in the recreated remote. This history must not be represented as the original from-scratch development sequence.

Planned verified stages:

1. Specification and provenance.
2. Autodiff engine with independent operation checks.
3. MLP/Adam and actual mask lifecycle.
4. Pruning, sparse export and training integration.
5. Fresh evaluation runs and generated evidence.
6. Final documentation and clean-checkout verification.

## Completed local verification

- Engine stage: 17 tests passed.
- MLP/Adam stage: 22 tests passed.
- Training/pruning stage: 29 tests passed.
- Additional direct-leaf backward fix and broadcast check: 31 tests passed.
- Fresh evaluation: all 30 paired held-out runs completed; raw results and plots regenerated here.

## Final submission polish

Pinned tested package versions, clarified recreated-remote provenance, expanded core operations into readable steps, added fixed-mask finite differences and changed integer removal budgets to round up. The expanded suite passes 33 tests. Experiments are regenerated after the budget change; RESULTS.md supersedes earlier measurements.

The user recreated sdm0p/tredence-aqua as an empty repository. Only this rebuild's staged history is published on main. The abandoned local integration attempt is retained on a local backup branch and is not published. The source commit above identifies the earlier implementation; that earlier history is not included in this recreated remote.
