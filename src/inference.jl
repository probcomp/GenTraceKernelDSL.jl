using LinearAlgebra

### Top-level inference algorithms ###

## SMCP3 Update
## See github.com/probcomp/GenSMCP3.jl for more details.
function run_smcp3_step(
    trace::Gen.Trace, new_target_args, target_argdiffs, update_constraint,
    k::Kernel, l::Kernel, k_args::Tuple, l_args::Tuple;
    check_are_inverses=false
)
    diff_config = DFD.DiffConfig()
    trace_token = TraceToken(trace, Gen.choicemap(), diff_config)

        # Run forward and compute proposed constraints, backward kernel choices, and jacobian correction.
    (proposed_update, backward_choices), forward_choices, forward_score = propose(
        k, (trace_token, k_args...), diff_config
    )
    jacobian_correction = compute_jacobian_correction(proposed_update, backward_choices)

    # Obtain new trace + model ratio
    new_model_trace, model_ratio, _, _ = Gen.update(
        trace, new_target_args, target_argdiffs,
        Gen.merge(undualize_choices(proposed_update), update_constraint)
    )

    # Assess backward kernel score
    (bwd_update, fwd_constraints), backward_score = assess(
        l, (new_model_trace, l_args...), undualize_choices(backward_choices)
    )

    if check_are_inverses
        check_round_trip(forward_choices, fwd_constraints, "Proposal")

        reconstructed_original_tr, _, _, _ = Gen.update(new_model_trace, Gen.get_args(trace),  map(x -> Gen.UnknownChange(), Gen.get_args(trace)), bwd_update)
        check_round_trip(forward_choices, fwd_constraints, "Proposal")
        check_round_trip(Gen.get_choices(trace), Gen.get_choices(reconstructed_original_tr), "Trace")
    end
   
    return new_model_trace, model_ratio + backward_score - forward_score + log(jacobian_correction)
end

Gen.@gen (static) function emptygf()
    return nothing
end
Gen.@load_generated_functions()

# This runs an SMC update, from an empty initial trace.
# The trace passed into the update will be an empty trace with no arguments, which returns nothing.
function run_initial_smcp3_step(
    target, target_args, constraint, k::Kernel, l::Kernel, k_args::Tuple, l_args::Tuple;
    check_are_inverses=false
)
    diff_config = DFD.DiffConfig()
    trace_token = TraceToken(Gen.simulate(emptygf, ()), Gen.choicemap(), diff_config)

    (proposed_update, backward_choices), forward_choices, forward_score = propose(
        k, (trace_token, k_args...), diff_config
    )
    jacobian_correction = compute_jacobian_correction(proposed_update, backward_choices)

    # Obtain new trace + model ratio
    new_model_trace, generate_ratio = Gen.generate(
        target, target_args,
        Gen.merge(undualize_choices(proposed_update), constraint)
    )

    # Assess backward kernel score
    (bwd_update, fwd_constraints), backward_score = assess(
        l, (new_model_trace, l_args...), undualize_choices(backward_choices)
    )

    @assert isempty(bwd_update) "Backward trace must be empty in an initial SMCP3 proposal."

    if check_are_inverses
        check_round_trip(forward_choices, fwd_constraints, "Proposal")
    end

    return new_model_trace, generate_ratio + backward_score - forward_score + log(jacobian_correction)
end

## Metropolis-Hastings (Involutive MCMC)
## TODO: arxiv link
function run_mcmc_kernel(trace::Gen.Trace, proposal::Kernel, other_args = ())
    diff_config = DynamicForwardDiff.DiffConfig()
    trace_token = TraceToken(trace, Gen.choicemap(), diff_config)
    
    (proposed_update, backward_choices), forward_choices, forward_score = propose(proposal, (trace_token, other_args...), diff_config)
    new_trace, model_log_weight, = Gen.update(trace, trace.args, ((Gen.NoChange() for _ in trace.args)...,), undualize_choices(proposed_update))
    _, backward_score = assess(proposal, (new_trace, other_args...), undualize_choices(backward_choices))

    jacobian_correction = compute_jacobian_correction(proposed_update, backward_choices)

    return new_trace, model_log_weight - forward_score + backward_score + log(jacobian_correction)
end

## Override the default Gen.metropolis_hastings function
## TODO: add support for `check` and `observations` kwargs
function Gen.metropolis_hastings(trace, proposal::Kernel)
    new_trace, alpha = run_mcmc_kernel(trace, proposal)
    if log(rand()) < alpha
        return new_trace, true
    else
        return trace, false
    end
end

### Sub-computations for inference algorithms ###
function accumulate_output_partials!(choices, outputs)
    for (k, v) in Gen.get_values_shallow(choices)
        if v isa DFD.Dual
            p = DFD.partials(v)
            !iszero(p) && push!(outputs, p)
        end
    end
    for (k, v) in Gen.get_submaps_shallow(choices)
        accumulate_output_partials!(v, outputs)
    end
end

function compute_jacobian_correction(proposed_update, backward_choices)
    continuous_inputs = Set()

    output_partials = []
    accumulate_output_partials!(proposed_update, output_partials)
    accumulate_output_partials!(backward_choices, output_partials)

    if isempty(output_partials)
        return 1.0
    end

    jacobian = hcat(output_partials...)

    # remove any rows which are all 0s -- these rows come from "read"s which none of the output "write"s depend upon
    trimmed_jacobian = hcat(
        (row for row in eachrow(jacobian) if !iszero(row))...
    )

    # each row contains everything related to a single read; each col, everything related to a single write
    if length(size(trimmed_jacobian)) == 1
        @assert length(trimmed_jacobian) == 1 "Jacobian Matrix is not square! Jacobian size = (1, $(length(trimmed_jacobian)))"
    else
        xsize, ysize = size(trimmed_jacobian)
        @assert xsize == ysize "Jacobian Matrix is not square! Jacobian size = $(size(trimmed_jacobian))"
    end

    abs(det(trimmed_jacobian))
end
