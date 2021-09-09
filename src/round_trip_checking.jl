function get_differences(c1, c2; is_approx=isapprox)
    in1not2 = []
    in2not1 = []
    diffval = []

    for (addr, sub1) in get_subtrees_shallow(c1)
        isempty(sub1) && continue
        sub2 = get_subtree(c2, addr)
        if isempty(sub2)
            push!(in1not2, addr)
        elseif sub1 isa Value && sub2 isa Value
            if !is_approx(get_value(sub1), get_value(sub2))
                push!(diffval, addr)
            end
        else
            (sub1not2, sub2not1, subdiffvals) = get_differences(sub1, sub2)
            for x in sub1not2
                push!(in1not2, addr => x)
            end
            for x in sub2not1
                push!(in2not1, addr => x)
            end
            for x in subdiffvals
                push!(diffval, addr => x)
            end
        end
    end
    for (addr, sub2) in get_subtrees_shallow(c2)
        isempty(sub2) && continue
        sub1 = get_subtree(c1, addr)
        if isempty(sub1)
            push!(in2not1, addr)
        end
    end

    return (in1not2, in2not1, diffval)
end

function errorlog_on_differences(before_rt, after_rt; is_approx=isapprox)
    (in1not2, in2not1, diffval) = get_differences(before_rt, after_rt; is_approx)
    strs = []
    for addr in in1not2
        push!(strs, "\n  Found before round trip but not after round trip: $addr")
    end
    for addr in in2not1
        push!(strs, "\n  Found after round trip but not before round trip: $addr")
    end
    for addr in diffval
        push!(strs, "\n  Different vals at: $addr. Value is $(before_rt[addr]) initially and $(after_rt[addr]) after round trip.")
    end

    if !isempty(strs)
        @error("differences: $(strs...)")
    else
        @error("No differences found!  Will log whole choicemaps.  isapprox(before_rt, after_rt) = $(isapprox(before_rt, after_rt)) | is_approx(before_rt, after_rt) = $(is_approx(before_rt, after_rt))")
        @error("Before RT:", before_rt) 
        @error("After RT:", after_rt)
    end
end
function check_round_trip(before_rt, after_rt, str; atol)
    is_approx = isnothing(atol) ? isapprox : (a, b) -> isapprox(a, b; atol)
    if !is_approx(before_rt, after_rt)
        @error("$str choices did not match after round trip!  Differences:")
        errorlog_on_differences(before_rt, after_rt; is_approx)
        error("transform round trip check failed")
    end
    return nothing
end