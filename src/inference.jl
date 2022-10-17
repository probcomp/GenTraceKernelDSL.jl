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

function run_mcmc_kernel(trace::Gen.Trace, k::MHProposal, other_args = ())
    diff_config = DynamicForwardDiff.DiffConfig()
    trace_token = TraceToken(trace, Gen.choicemap(), diff_config)
    
    (proposed_update, backward_choices), forward_choices, forward_score = propose(k.proposal, (trace_token, other_args...), diff_config)
    new_trace, model_log_weight, = Gen.update(trace, trace.args, ((Gen.NoChange() for _ in trace.args)...,), undualize_choices(proposed_update))
    _, backward_score = assess(k.proposal, (new_trace, other_args...), undualize_choices(backward_choices))

    jacobian_correction = compute_jacobian_correction(proposed_update, backward_choices)

    return new_trace, model_log_weight - forward_score + backward_score + log(jacobian_correction)
end

function run_mh(trace::Gen.Trace, k::MHProposal, other_args=())
    new_trace, alpha = run_mcmc_kernel(trace, k, other_args)
    if log(rand()) < alpha
        return new_trace, true
    else
        return trace, false
    end
end

function run_smc_step(trace::Gen.Trace, k::SMCStep, fwd_args=(), bwd_args=())
    diff_config = DFD.DiffConfig()
    trace_token = TraceToken(trace, Gen.choicemap(), diff_config)

    # Run forward and compute proposed constraints, backward kernel choices, and jacobian correction.
    (proposed_update, backward_choices), forward_choices, forward_score = propose(k.forward, (trace_token, fwd_args...), diff_config)
    jacobian_correction = compute_jacobian_correction(proposed_update, backward_choices)

    # Obtain new trace + model ratio
    new_model_trace, model_ratio, _, _ = Gen.update(
        trace, k.target_args, map(x -> Gen.UnknownChange(), k.target_args),
        Gen.merge(undualize_choices(proposed_update), k.observations)
    )
    
    # Assess backward kernel score
    _, backward_score = assess(k.backward, (new_model_trace, bwd_args...), undualize_choices(backward_choices))

    return new_model_trace, model_ratio + backward_score - forward_score + log(jacobian_correction)
end

# Add methods to `particle_filter_step!` for SMCP³
function Gen.particle_filter_step!(state::Gen.ParticleFilterState{U}, step::SMCStep, fwd_args::Tuple, bwd_args::Tuple) where {U}
    num_particles = length(state.traces)
    log_incremental_weights = Vector{Float64}(undef, num_particles) 
    for i=1:num_particles
        (state.new_traces[i], log_weight) = run_smc_step(state.traces[i], step, fwd_args, bwd_args)
        log_incremental_weights[i] = log_weight
        state.log_weights[i] += log_weight
    end
    # swap references
    tmp = state.traces
    state.traces = state.new_traces
    state.new_traces = tmp

    return (log_incremental_weights,)
end
function Gen.particle_filter_step!(state::Gen.ParticleFilterState{U}, new_args::Tuple, argdiffs::Tuple,
    observations::Gen.ChoiceMap, k::Kernel, l::Kernel, fwd_args::Tuple, bwd_args::Tuple) where {U}
    step = SMCStep(k, l, Gen.get_gen_fn(first(state.traces)), new_args, observations)
    return Gen.particle_filter_step!(state, step, fwd_args, bwd_args)
end

export MHProposal, SMCStep, run_mh, run_smc_step