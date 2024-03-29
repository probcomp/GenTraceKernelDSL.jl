# Like a generative function, but with a slightly different interface
#   * When calling a Gen GF, only the ChoiceMap is available, not the return value.
#   * When calling a Kernel, only the return value is available, not the ChoiceMap.
struct Kernel
    fn :: Function
end

# A wrapper for a Kernel which lets other kernels
# access its choices.
struct TracedKernel
    kernel :: Kernel
end

# Modified from Gen
function desugar_tildes(expr, sample_func)
    MacroTools.postwalk(expr) do e
        # Replace tilde statements with :gentrace expressions
        if MacroTools.@capture(e, {*} ~ f_(args__))
            :($(sample_func)($f, ($(args...),)))
        elseif MacroTools.@capture(e, {addr_} ~ f_(args__))
            :($(sample_func)($f, ($(args...),), $addr))
        elseif MacroTools.@capture(e, lhs_Symbol ~ f_(args__))
            :($lhs = $sample_func($f, ($(args...),), $(Meta.quot(lhs))))
        elseif MacroTools.@capture(e, lhs_ ~ rhs_call)
            error("Syntax error: Invalid left-hand side: $(e)." *
                  "Only a variable or address can appear on the left of a `~`.")
        elseif MacroTools.@capture(e, lhs_ ~ rhs_)
            error("Syntax error: Invalid right-hand side in: $(e)")
        else
            e
        end
    end
end

# Other question is, what about 'trace token'?
macro kernel(fndef)
    func = splitdef(fndef)
    sample_func = gensym("sample")
    func[:args] = [sample_func, func[:args]...]
    func[:body] = desugar_tildes(func[:body], sample_func)
    
    
    name = func[:name]
    delete!(func, :name)
    func_expr = MacroTools.combinedef(func)

    quote
        $(esc(name)) = Kernel($(esc(func_expr)))
    end
end

# Returns outputs as dual numbers, as well as a ChoiceMap of all choices made, and a score.
function propose(kernel::Kernel, args, diff_config = DynamicForwardDiff.DiffConfig())
    choices = Gen.choicemap()
    score   = 0.0

    # Continuous choices
    function sample(f::Gen.Distribution{<: Union{Float64, AbstractArray{Float64}}}, args, addr)
        arg_values = DFD.value.(args)
        x = Gen.random(f, arg_values...)
        choices[addr] = x
        score += Gen.logpdf(f, x, arg_values...)
        dualized = DFD.new_dual(diff_config, x)
        return dualized
    end

    # Discrete choices
    function sample(f::Gen.Distribution, args, addr)
        arg_values = DFD.value.(args)
        x = Gen.random(f, arg_values...)
        choices[addr] = x
        score += Gen.logpdf(f, x, arg_values...)
        return x
    end

    # Gen Generative Functions
    function sample(f::Gen.GenerativeFunction, args, addr = nothing)
        trace = Gen.simulate(f, DFD.value.(args))
        submap = Gen.get_choices(trace)
        score += Gen.get_score(trace)
        if isnothing(addr)
            choices = merge(choices, submap)
        else
            Gen.set_submap!(choices, addr, submap)
        end
        return TraceToken(trace, Gen.choicemap(), diff_config)
    end

    # Other kernels
    function sample(f::Kernel, args, addr = nothing)
        ret, submap, subscore = propose(f, args, diff_config)
        score += subscore
        if isnothing(addr)
            choices = merge(choices, submap)
        else
            Gen.set_submap!(choices, addr, submap)
        end
        return ret
    end
    function sample(f::TracedKernel, args, addr = nothing)
        ret, submap, subscore = propose(f.kernel, args, diff_config)
        score += subscore
        if isnothing(addr)
            choices = merge(choices, submap)
        else
            Gen.set_submap!(choices, addr, submap)
        end
        return (ret, submap, subscore)
    end

    ret = kernel.fn(sample, args...)

    return ret, choices, score
end

# Given a ChoiceMap, assess probability and return retval
function assess(kernel::Kernel, args, choices::Gen.ChoiceMap)
    score = 0.0
    function increment_score(f::Gen.Distribution, args, addr)
        score += Gen.logpdf(f, choices[addr], args...)
        return choices[addr]
    end
    function increment_score(f::Gen.GenerativeFunction, args, addr = nothing)
        submap = isnothing(addr) ? choices : Gen.get_submap(choices, addr)
        trace, subscore = Gen.generate(f, args, submap)
        score += subscore
        return trace
    end
    function increment_score(f::Kernel, args, addr = nothing)
        submap = isnothing(addr) ? choices : Gen.get_submap(choices, addr)
        retval, subscore = assess(f, args, submap)
        score += subscore
        return retval
    end
    function increment_score(f::TracedKernel, args, addr = nothing)
        submap = isnothing(addr) ? choices : Gen.get_submap(choices, addr)
        retval, subscore = assess(f.kernel, args, submap)
        score += subscore
        return (retval, submap, subscore)
    end

    ret = kernel.fn(increment_score, args...)
    return ret, score
end

export @kernel, Kernel