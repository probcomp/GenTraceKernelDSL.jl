# GenTraceKernelDSL.jl

This package provides a DSL for constructing _trace kernels_, stochastic
maps between the traces of [Gen](https://github.com/probcomp/Gen.jl) generative
functions, for use as (generalized) Metropolis-Hastings or SMC proposals.

This package can be viewed as a refactoring of [Gen's Trace Translator functionality](https://github.com/probcomp/Gen.jl/blob/a96a77991e0e43f208272e9241c8f2434ffdedbf/docs/src/ref/trace_translators.md),
described in [Marco Cusumano-Towner's thesis](https://www.mct.dev/assets/mct-thesis.pdf) and also in the arXiv preprint
[Automating Involutive MCMC using Probabilistic and Differentiable Programming](https://arxiv.org/abs/2007.09871).
Unlike Gen's trace transform DSL, this package does not enforce separation of the "probablistic" and "differentiable" components of a trace translator: users may freely mix sampling with deterministic transformations to describe arbitrary stochastic transformations.

## DSL

A kernel function is declared using the `@kernel` macro.
The kernel's body may contain deterministic Julia code, as well as `~` expressions,
familiar from Gen:

* `{:x} ~ dist(args)` samples from a Gen distribution at address `:x`
* `{:x} ~ gen_fn(args)` samples from a Gen generative function at address `:x`, **and evaluates to the _trace_ of the function, rather than its return value**
* `{:x} ~ kernel_fn(args)` calls another `@kernel`-defined function at address `:x`, **and evaluates to its return value.**

As in Gen, `x = {:x} ~ f()` can be shortened to `x ~ f()`, and—for generative function or kernel calls—the `{*} ~ f()` syntax can be used to splice the choices made by `f` into the "top level" of the caller's choicemap.

Kernels intended for use as MH proposals should accept a current trace as their first argument, and return a Tuple of: (1) a `ChoiceMap` of proposed values to update in the trace, and (2) a `ChoiceMap` specifying a reverse move. 

Kernels inteded for use as SMC proposals should be written in pairs: a forward and backward kernel. The forward (backward) kernel should accept a previous (subsequent) model trace as its first argument, and return a Tuple containing: (1) a `ChoiceMap` specifying a proposed next (previous) model state, and (2) a `ChoiceMap` of the backward (forward) kernel that would recover the previous (subsequent) model state.

For example, here is what [Gen's example split-merge proposal](https://github.com/probcomp/Gen.jl/blob/master/examples/involutive_mcmc/involution_mh_minimal_example.jl) looks like written in the DSL:

```julia
@kernel function split_merge_proposal(trace)
    if trace[:z]
        # Currently two means, switch to one
        m, u  = merge_mean(trace[:m1], trace[:m2])
        return choicemap(:z => false, :m => m), choicemap(:u => u)
    else
        # Currently one mean, switch to two
        u ~ uniform_continuous(0, 1)
        m1, m2 = split_mean(trace[:m], u)
        return choicemap(:z => true, :m1 => m1, :m2 => m2), choicemap()
    end
end
```

Kernels can be wrapped in `MHProposal` or `SMCStep` objects and passed to `mh(trace, proposal, args)` or `run_smc_step(trace, step, fwd_args, bwd_args)` respectively.


See `example.jl` for a full example.