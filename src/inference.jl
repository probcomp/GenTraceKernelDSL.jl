using LinearAlgebra

# General SMC Step -- does not exploit incremental computation when target and previous model 
# are the same generative function (just with different args / obs).
struct SMCStep
    forward      :: Kernel  # Trace -> Tuple{ChoiceMap, ChoiceMap} -- of target and backward
    backward     :: Kernel  # Trace -> Tuple{ChoiceMap, ChoiceMap} -- of target and forward
    target       :: Gen.GenerativeFunction
    target_args  :: Tuple
    observations :: Gen.ChoiceMap
end

struct MHProposal
    proposal :: Kernel # Trace -> Tuple{ChoiceMap, ChoiceMap} -- of target and self
end

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
        return 0.0
    end

    # TODO: Improve this to identify copies, etc.
    n_inputs = first(output_partials).length[]
    @assert length(output_partials) == n_inputs
    jacobian = vcat(transpose.(output_partials)...)
    abs(det(jacobian))
end

function run_mcmc_kernel(trace::Gen.Trace, k::MHProposal, other_args = ())
    diff_config = DynamicForwardDiff.DiffConfig()
    trace_token = TraceToken(trace, Gen.choicemap(), diff_config)
    
    (proposed_update, backward_choices), forward_choices, forward_score = propose(k.proposal, (trace_token, other_args...), diff_config)
    new_trace, model_log_weight, = Gen.update(trace, trace.args, ((Gen.NoChange() for _ in trace.args)...,), undualize_choices(proposed_update))
    _, backward_score = assess(k.proposal, (new_trace, other_args...), undualize_choices(backward_choices))

    jacobian_correction = compute_jacobian_correction(proposed_update, backward_choices)

    return new_trace, model_log_weight - forward_score + backward_score
end

function run_mh(trace::Gen.Trace, k::MHProposal, other_args=())
    new_trace, alpha = run_mcmc_kernel(trace, k, other_args)
    if log(rand()) < alpha
        return new_trace, true
    else
        return trace, false
    end
end


# Untested:
function run_smc_step(trace::Gen.Trace, k::SMCStep, fwd_args=(), bwd_args=())
    diff_config = DFD.DiffConfig()
    trace_token = TraceToken(trace, Gen.choicemap(), diff_config)

    # Run forward and compute proposed constraints, backward kernel choices, and jacobian correction.
    (proposed_update, backward_choices), forward_choices, forward_score = propose(k.forward, (trace_token, fwd_args...), diff_config)
    jacobian_correction = compute_jacobian_correction(proposed_update, backward_choices)

    # Obtain new trace + new model score
    constraints = merge(undualize_choices(proposed_update), k.observations)
    (new_model_trace, new_model_score) = Gen.generate(k.target, k.target_args, constraints)
    
    # Assess backward kernel score
    _, backward_score = assess(k.backward, (new_model_trace, bwd_args...), undualize_choices(backward_choices))

    # Compute & return full log weight
    old_model_score = get_score(trace)
    return new_model_trace, new_model_score - old_model_score + backward_score - forward_score
end

export MHProposal, SMCStep, run_mh, run_smc_step