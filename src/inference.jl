using LinearAlgebra

######
# MH #
###### 
struct MHProposal
    proposal :: Kernel # Trace -> Tuple{ChoiceMap, ChoiceMap} -- of target and self
end

function dualized_values(spec::Gen.AddressTree{<:Union{Value, Selection}})
    (v for v in all_values_deep(spec) if v isa DFD.Dual)
end
function accumulate_output_partials!(spec::Gen.UpdateSpec, outputs)
    for v in dualized_values(spec)
        p = DFD.partials(v)
        if !iszero(p)
            push!(outputs, p)
        end
    end
end

function accumulate_output_partials!(choices::Gen.AddressTree{<:Union{Value, Selection}}, outputs)
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
    # n_inputs = first(output_partials).length[]

    # unique_reads = Set(keys(p.values) for p in output_partials)
    # if length(output_partials) != n_inputs
    #     display(output_partials)
    #     error("Jacobian Matrix isn't square! n_inputs = $n_inputs; length(output_partials) = $(length(output_partials))")
    # end
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

unpack_backward_spec(backward_spec::Tuple) = backward_spec
unpack_backward_spec(backward_spec) = (backward_spec, EmptySelection())

include("round_trip_checking.jl")

function incorporate_regenerate_constraints!(updatespec::Gen.DynamicAddressTree{<:Union{Value, SelectionLeaf}},
    trace_choices::Gen.ChoiceMap, reverse_regenerated::Gen.Selection
)
    for (addr, subtree) in get_subtrees_shallow(reverse_regenerated)
        set_subtree!(updatespec, addr,
            incorporate_regenerate_constraints!(get_subtree(updatespec, addr), get_subtree(trace_choices, addr), subtree)
        )
    end
    return updatespec
end
function incorporate_regenerate_constraints!(updatespec::Gen.DynamicAddressTree{<:Union{Value, SelectionLeaf}},
    trace_choices::Gen.ChoiceMap, reverse_regenerated::Gen.SelectionLeaf
)
    # this case is a bit tricky, since we can't enumerate all the subtrees of `reverse_regenerated`    
    error("Not implemented.")
end
incorporate_regenerate_constraints!(::Gen.AddressTree, trace_choices::Gen.ChoiceMap, ::AllSelection) = trace_choices
incorporate_regenerate_constraints!(spec::Gen.AddressTree{<:Union{Value, SelectionLeaf}}, ::Gen.ChoiceMap, ::EmptySelection) = spec
incorporate_regenerate_constraints!(spec::Gen.DynamicAddressTree{<:Union{Value, SelectionLeaf}}, ::Gen.ChoiceMap, ::EmptySelection) = spec
incorporate_regenerate_constraints!(t::EmptyAddressTree, ::ChoiceMap, ::EmptySelection) = t

function run_mcmc_kernel(trace::Gen.Trace, k::MHProposal, other_args = ();
    do_round_trip_check=false,
    log_fwd_choices=false,
    log_proposed_update=false,
    log_bwd_choices=false,
    log_bwd_regenerated=false,
    log_roundtrip_update_spec=false,
    log_roundtrip_bwd_choices=false,
    log_roundtrip_bwd_regenerated=false,
    log_new_trace_choicemap=false,
    log_roundtrip_update_with_regen=false,
    roundtrip_atol=nothing
)
    diff_config = DynamicForwardDiff.DiffConfig()
    trace_token = TraceToken(trace, Gen.choicemap(), diff_config)
    
    (proposed_update, backward_spec), forward_choices, forward_score = propose(k.proposal, (trace_token, other_args...), diff_config)
    (backward_choices, reverse_regenerated) = unpack_backward_spec(backward_spec)

    if log_fwd_choices
        @info("Forward proposal choices: $forward_choices")
    end
    if log_proposed_update
        @info("Proposed update: $proposed_update")
    end
    if log_bwd_choices
        @info("Specified backward choices: $backward_choices")
    end
    if log_bwd_regenerated
        @info("Specified addresses for regeneration during the backward move: $reverse_regenerated")
    end

    new_trace, model_log_weight, = Gen.update(trace, get_args(trace), ((Gen.NoChange() for _ in get_args(trace))...,),
        undualize(proposed_update), invert(reverse_regenerated)
    )

    _, backward_score = assess(k.proposal, (new_trace, other_args...), undualize(backward_choices))

    if log_new_trace_choicemap
        @info("New trace choicemap: $(get_choices(new_trace))")
    end

    jacobian_correction = compute_jacobian_correction(proposed_update, backward_choices)

    if do_round_trip_check
        new_trace_token = TraceToken(new_trace, Gen.choicemap(), DynamicForwardDiff.DiffConfig())
        (roundtrip_update, roundtrip_bwd_spec) = run_with_choices(k.proposal, (new_trace_token, other_args...), backward_choices)
        roundtrip_update = undualize(roundtrip_update)
        (roundtrip_bwd_choices, roundtrip_bwd_regenerated) = unpack_backward_spec(roundtrip_bwd_spec)

        # TODO: logging in the round-trip
        if log_roundtrip_update_spec
            @info("Round-trip update spec: $roundtrip_update")
        end
        if log_roundtrip_bwd_choices
            @info("Round-trip backward move specification: $roundtrip_bwd_choices")
        end
        if log_roundtrip_bwd_regenerated
            @info("Addresses the round-trip specified are regenerated in its bwd move: $roundtrip_bwd_regenerated")
        end

        roundtrip_update = incorporate_regenerate_constraints!(roundtrip_update, get_choices(trace), reverse_regenerated)

        if log_roundtrip_update_with_regen
            @info("Roundtrip update with regen constraints: $roundtrip_update")
        end

        roundtrip_trace, _, _, _ = Gen.update(new_trace, get_args(new_trace), ((Gen.NoChange() for _ in get_args(trace))...,), roundtrip_update, EmptyAddressTree())
        
        check_round_trip(get_choices(trace), get_choices(roundtrip_trace), "Trace"; atol=roundtrip_atol)
        check_round_trip(forward_choices, undualize(roundtrip_bwd_choices), "Proposal"; atol=roundtrip_atol)

        # TODO: should we have an explicit check for whether the regeneration reversal specificaions were valid?
        # or do we catch enough of those errors in the choicemap roundtrip checks?
     end

    # println("model_ratio = $model_log_weight | fwd = $(-forward_score) | bwd = $backward_score | sum = $(model_log_weight - forward_score + backward_score)")
    return new_trace, model_log_weight - forward_score + backward_score
end

function Gen.mh(trace::Gen.Trace, k::MHProposal, other_args=();
    check=false, observations=EmptyChoiceMap(),
    log_fwd_choices=false,
    log_proposed_update=false,
    log_bwd_choices=false,
    log_bwd_regenerated=false,
    log_roundtrip_update_spec=false,
    log_roundtrip_bwd_choices=false,
    log_roundtrip_bwd_regenerated=false,
    log_new_trace_choicemap=false,
    log_roundtrip_update_with_regen=false,
    roundtrip_atol=nothing
)
    new_trace, alpha = run_mcmc_kernel(trace, k, other_args;
        do_round_trip_check=check, log_fwd_choices, log_proposed_update, log_bwd_choices, log_bwd_regenerated,
        log_new_trace_choicemap, log_roundtrip_update_spec, log_roundtrip_bwd_choices, log_roundtrip_bwd_regenerated, log_roundtrip_update_with_regen, roundtrip_atol
    )
    if log(rand()) < alpha
        if check && !isempty(observations)
            Gen.check_observations(get_choices(new_trace), observations)
        end
        return new_trace, true
    else
        return trace, false
    end
end
Gen.mh(trace::Gen.Trace, k::Kernel, args...; kwargs...) = Gen.mh(trace, MHProposal(k), args...; kwargs...)

#######
# SMC #  TODO!!
#######

# General SMC Step -- does not exploit incremental computation when target and previous model 
# are the same generative function (just with different args / obs).
# struct SMCStep
#     forward      :: Kernel  # Trace -> Tuple{ChoiceMap, ChoiceMap} -- of target and backward
#     backward     :: Kernel  # Trace -> Tuple{ChoiceMap, ChoiceMap} -- of target and forward
#     target       :: Gen.GenerativeFunction
#     target_args  :: Tuple
#     observations :: Gen.ChoiceMap
# end

# function run_smc_step(trace::Gen.Trace, k::SMCStep, fwd_args=(), bwd_args=())
#     diff_config = DFD.DiffConfig()
#     trace_token = TraceToken(trace, Gen.choicemap(), diff_config)

#     # Run forward and compute proposed constraints, backward kernel choices, and jacobian correction.
#     (proposed_update, backward_choices), forward_choices, forward_score = propose(k.forward, (trace_token, fwd_args...), diff_config)
#     jacobian_correction = compute_jacobian_correction(proposed_update, backward_choices)

#     # Obtain new trace + new model score
#     constraints = merge(undualize_choices(proposed_update), k.observations)
#     (new_model_trace, new_model_score) = Gen.generate(k.target, k.target_args, constraints)
    
#     # Assess backward kernel score
#     _, backward_score = assess(k.backward, (new_model_trace, bwd_args...), undualize_choices(backward_choices))

#     # Compute & return full log weight
#     old_model_score = get_score(trace)
#     return new_model_trace, new_model_score - old_model_score + backward_score - forward_score
# end

export MHProposal, run_mh