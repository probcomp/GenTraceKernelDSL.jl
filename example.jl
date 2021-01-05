using Gen
using GenTraceKernelDSL
using Random

# From https://github.com/probcomp/Gen.jl/blob/a96a77991e0e43f208272e9241c8f2434ffdedbf/examples/involutive_mcmc/involution_mh_minimal_example.jl
@gen function model()
    if ({:z} ~ bernoulli(0.5))
        m1 = ({:m1} ~ gamma(1, 1))
        m2 = ({:m2} ~ gamma(1, 1))
    else
        m = ({:m} ~ gamma(1, 1))
        (m1, m2) = (m, m)
    end
    {:y1} ~ normal(m1, 0.1)
    {:y2} ~ normal(m2, 0.1)
end

# From https://github.com/probcomp/Gen.jl/blob/a96a77991e0e43f208272e9241c8f2434ffdedbf/examples/involutive_mcmc/involution_mh_minimal_example.jl
@gen function mean_random_walk_proposal(trace)
    if trace[:z]
        {:m1} ~ normal(trace[:m1], 0.1)
        {:m2} ~ normal(trace[:m2], 0.1)
    else
        {:m} ~ normal(trace[:m], 0.1)
    end
end

# From https://github.com/probcomp/Gen.jl/blob/a96a77991e0e43f208272e9241c8f2434ffdedbf/examples/involutive_mcmc/involution_mh_minimal_example.jl
function merge_mean(m1, m2)
    m = sqrt(m1 * m2)
    u = m1 / (m1 + m2)
    (m, u)
end

# From https://github.com/probcomp/Gen.jl/blob/a96a77991e0e43f208272e9241c8f2434ffdedbf/examples/involutive_mcmc/involution_mh_minimal_example.jl
function split_mean(m, u)
    m1 = m * sqrt((u / (1 - u)))
    m2 = m * sqrt(((1 - u) / u))
    (m1, m2)
end

# New: a split-merge proposal defined in a single 
# function body combining stochastic and deterministic
# code.
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

# From https://github.com/probcomp/Gen.jl/blob/a96a77991e0e43f208272e9241c8f2434ffdedbf/examples/involutive_mcmc/involution_mh_minimal_example.jl
function do_inference_simple(y1, y2)
    trace, = generate(model, (), choicemap((:y1, y1), (:y2, y2), (:z, false), (:m, 1.2)))
    zs = Bool[]
    m = Float64[]
    m1 = Float64[]
    m2 = Float64[]
    for iter=1:100
        trace, = mh(trace, select(:z))
        trace, = mh(trace, mean_random_walk_proposal, ())
        push!(zs, trace[:z])
        push!(m, trace[:z] ? NaN : trace[:m])
        push!(m1, trace[:z] ? trace[:m1] : NaN)
        push!(m2, trace[:z] ? trace[:m2] : NaN)
    end
    (zs, m, m1, m2)
end

# From https://github.com/probcomp/Gen.jl/blob/a96a77991e0e43f208272e9241c8f2434ffdedbf/examples/involutive_mcmc/involution_mh_minimal_example.jl
function do_inference_rjmcmc(y1, y2)
    trace, = generate(model, (), choicemap((:y1, y1), (:y2, y2), (:z, false), (:m, 1.2)))
    zs = Bool[]
    m = Float64[]
    m1 = Float64[]
    m2 = Float64[]
    for iter=1:100
        trace, = run_mh(trace, MHProposal(split_merge_proposal))
        trace, = mh(trace, mean_random_walk_proposal, ())
        push!(zs, trace[:z])
        push!(m, trace[:z] ? NaN : trace[:m])
        push!(m1, trace[:z] ? trace[:m1] : NaN)
        push!(m2, trace[:z] ? trace[:m2] : NaN)
    end
    (zs, m, m1, m2)
end

using Plots

# From https://github.com/probcomp/Gen.jl/blob/a96a77991e0e43f208272e9241c8f2434ffdedbf/examples/involutive_mcmc/involution_mh_minimal_example.jl
function make_plots()
    Random.seed!(3)
    
    y1, y2 = (1.0, 1.3)
    (zs, m, m1, m2) = @time do_inference_rjmcmc(y1, y2)
    p1 = plot(title="Involution MH (RJMCMC)", m, label="m")
    plot!(m1, label="m1")
    plot!(m2, label="m2")
    ylims!(0.5, 1.5)
    
    p2 = plot(zs, label="z", color="black")
    xlabel!("# MCMC moves")
    yticks!([0, 1], ["F", "T"])
    ylims!(-0.1, 1.1)
    
    y1, y2 = (1.0, 1.3)
    (zs, m, m1, m2) = @time do_inference_simple(y1, y2)
    p3 = plot(title="Selection MH", m, label="m")
    plot!(m1, label="m1")
    plot!(m2, label="m2")
    ylims!(0.5, 1.5)
    
    p4 = plot(zs, label="z", color="black")
    xlabel!("# MCMC moves")
    yticks!([0, 1], ["F", "T"])
    ylims!(-0.1, 1.1)
    
    plot(p1, p3, p2, p4)
    savefig("rjmcmc.png")
end

make_plots()