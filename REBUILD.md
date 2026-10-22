# Provenance and rebuild log

Source: sdm0p/tredence-aqua, commit 6999a8605b85cbb6434936d291cf24f4465a2e59.

This is a transparent staged migration/rebuild with additional verification. Source from the first implementation is reused. Commits use actual current timestamps; the original repository and its history remain intact. This history must not be represented as the original from-scratch development sequence.

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

This rebuild will replace the contents of the existing sdm0p/tredence-aqua repository through a merge preserving both histories, at the user's request. No forced history replacement is required.
