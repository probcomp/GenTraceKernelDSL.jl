struct TraceToken
    trace        :: Gen.Trace
    dual_choices :: Gen.ChoiceMap
    diff_config  :: DynamicForwardDiff.DiffConfig # tag and reference to int
end

@inline function Gen.get_args(t::TraceToken)
    Gen.get_args(t.trace)
end

function Gen.get_score(t::TraceToken)
    @warn "GenTraceKernelDSL's calculated weights may be incorrect if proposals depend continuously on result of `get_score`."
    return Gen.get_score(t.trace)
end

function Gen.get_retval(t::TraceToken)
    @warn "GenTraceKernelDSL's calculated weights may be incorrect if proposals depend continuously on result of `get_retval`."
    return Gen.get_retval(t.trace)
end

@inline function Gen.get_gen_fn(t::TraceToken)
    Gen.get_gen_fn(t.trace)
end

@inline function dualize_value(v::Union{Float64, AbstractArray{<: Float64}}, config)
    DFD.new_dual(config, v)
end
@inline dualize_value(v, config) = v

function Base.getindex(t::TraceToken, addr)
    if Gen.has_value(t.dual_choices, addr)
        return t.dual_choices[addr]
    end
    trace = t.trace
    v = dualize_value(trace[addr], t.diff_config)
    t.dual_choices[addr] = v
    return v
end

Base.getindex(trace::TraceToken) = Gen.get_retval(trace)


function dualize_choicemap!(c::Gen.ChoiceMap, dual_choices::Gen.ChoiceMap, cfg::DFD.DiffConfig)
    for (key, value) in Gen.get_values_shallow(c)
        if !has_value(dual_choices, key)
            dual_choices[key] = dualize_value(value, cfg)
        end
    end
    for (key, value) in Gen.get_submaps_shallow(c)
        if has_submap(dual_choices, key)
            submap = get_submap(dual_choices, key)
        else
            submap = choicemap()
            Gen.set_submap!(dual_choices, key, submap)
        end
        dualize_choicemap!(Gen.get_submap(c, key), submap, cfg)
    end
end

function undualize_choices(c::Gen.ChoiceMap)
    undual = Gen.choicemap()
    for (key, v) in Gen.get_values_shallow(c)
        undual[key] = DFD.value(v)
    end
    for (key, value) in Gen.get_submaps_shallow(c)
        Gen.set_submap!(undual, key, undualize_choices(Gen.get_submap(c, key)))
    end
    return undual
end

function Gen.get_choices(t::TraceToken)
    choices = Gen.get_choices(t.trace)
    dualize_choicemap!(choices, t.dual_choices, cfg)
    return t.dual_choices
end
