module GenTraceKernelDSL

import Gen
using DynamicForwardDiff
using MacroTools

const DFD = DynamicForwardDiff

include("kernel_dsl.jl")
include("trace_token.jl")
include("inference.jl")
include("round_trip_checking.jl")

end # module
