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

global retval_warning_given = false
function Gen.get_retval(t::TraceToken)
    if !retval_warning_given
        @warn "GenTraceKernelDSL's calculated weights may be incorrect if proposals depend continuously on result of `get_retval`.\nThis warning will not be given again."
        global retval_warning_given = true
    end
    return Gen.get_retval(t.trace)
end

# for users who know what they are doing:
get_retval_nowarning(t::TraceToken) = Gen.get_retval(t.trace)
get_undualed(t::TraceToken, addr) = t.trace[addr]
get_undualed(trace::Gen.Trace, addr) = trace[addr]

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


function dualize!(c::Gen.ChoiceMap, dual_choices::Gen.ChoiceMap, cfg::DFD.DiffConfig)
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
        dualize!(Gen.get_submap(c, key), submap, cfg)
    end
end

undualize(v::Gen.Value) = Value(DFD.value(get_value(v)))
undualize(s::Gen.AddressTree{<:SelectionLeaf}) = s
undualize(::EmptyAddressTree) = EmptyAddressTree()
function undualize(c::Gen.AddressTree{<:Union{Value, SelectionLeaf}})
    undual = c isa Gen.ChoiceMap ? Gen.choicemap() : Gen.regenchoicemap()
    for (addr, subtree) in Gen.get_subtrees_shallow(c)
        Gen.set_subtree!(undual, addr, undualize(subtree))
    end
    return undual
end

function Gen.get_choices(t::TraceToken)
    choices = Gen.get_choices(t.trace)
    dualize!(choices, t.dual_choices, cfg)
    return t.dual_choices
end
