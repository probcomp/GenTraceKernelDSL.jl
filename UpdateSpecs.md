# Kernel DSL with Update Specs

In this fork, I'm adding support for:
1. Regeneration during MH updates
2. Using arbitrary `UpdateSpec`s

## Generalized kernel-writing interface

The kernel function now returns either:
1. A tuple `(updatespec, backward_choices)` where `updatespec` may be any updatespec accepted by the generative function producing the trace the kernel will update.  If this signature is used, the updatespec must deterministically specify an update. (Ie. it must not regenerate any values.)
2. A tuple `(updatespec, backward_choices, reverse_regenerated)`.  `reverse_regenerated` is a selection of all the model addresses which will be regenerated during the reverse move.

## Interface for writing compatible UpdateSpec types

For an `UpdateSpecLeaf` subtype to be compatible with the kernel DSL, it must implemenent
`undualize` and `incorporate_regenerate_constraints`.

`undualize(updatespec)` should convert any value within the spec from a `Dual` value to a regular value.  (`undualize` can be
called on a choicemap to help with this.)

`incorporate_regenerate_constraints(updatespec, trace_choices::ChoiceMap, reverse_regenerated::Selection)` should return an update spec equivalent to
`updatespec`, but with all the addresses in `reverse_regenerated` constrained to take the values they take in `trace_choices`.
It may mutate the given `updatespec` (and then re-return it).