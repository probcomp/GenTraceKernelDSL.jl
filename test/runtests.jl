module GenTraceKernelDSLTests
using Test
using Gen
using GenTraceKernelDSL

### Example from `example.jl` ###

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
function do_inference_rjmcmc(y1, y2)
    trace, = generate(model, (), choicemap((:y1, y1), (:y2, y2), (:z, false), (:m, 1.2)))
    zs = Bool[]
    m = Float64[]
    m1 = Float64[]
    m2 = Float64[]
    for iter=1:100
        trace, = Gen.mh(trace, MHProposal(split_merge_proposal))
        trace, = mh(trace, mean_random_walk_proposal, ())
        push!(zs, trace[:z])
        push!(m, trace[:z] ? NaN : trace[:m])
        push!(m1, trace[:z] ? trace[:m1] : NaN)
        push!(m2, trace[:z] ? trace[:m2] : NaN)
    end
    (zs, m, m1, m2)
end

# test that no errors occur!
@testset "simple usage test" begin
    do_inference_rjmcmc(2, 5)
end

# TODO: acceptance probability check?

@testset "roundtrip check" begin
    # kernels which do not give correct reverse moves
    @kernel function invalid_split_merge_proposal1(trace)
        if trace[:z]
            # Currently two means, switch to one
            m, u  = merge_mean(trace[:m1], trace[:m2])
            return choicemap(:z => false, :m => m + 1), choicemap(:u => u)
        else
            # Currently one mean, switch to two
            u ~ uniform_continuous(0, 1)
            m1, m2 = split_mean(trace[:m], u)
            return choicemap(:z => false, :m1 => m1, :m2 => m2), choicemap()
        end
    end
    @kernel function invalid_split_merge_proposal2(trace)
        if trace[:z]
            # Currently two means, switch to one
            m, u  = merge_mean(trace[:m1], trace[:m2])
            return choicemap(:z => false, :m => m), choicemap(:u => u - 4)
        else
            # Currently one mean, switch to two
            u ~ uniform_continuous(0, 1)
            m1, m2 = split_mean(trace[:m], u)
            return choicemap(:z => true, :m1 => m1, :m2 => m2), choicemap(:u => 1)
        end
    end

    trace, = generate(model, (), choicemap((:y1, 1), (:y2, 1), (:z, false), (:m, 1.2)))
    @test_throws Exception Gen.mh(trace, MHProposal(invalid_split_merge_proposal1), check=true)
    @test_throws Exception Gen.mh(trace, MHProposal(invalid_split_merge_proposal2), check=true)

end

@testset "observations check" begin
    trace, = generate(model, (), choicemap((:y1, 5), (:y2, 1), (:z, false), (:m, 1.2)))
    @test_throws Exception Gen.mh(trace, MHProposal(split_merge_proposal); check=true,
        observations=choicemap((:y1, 1))
    )
end

end