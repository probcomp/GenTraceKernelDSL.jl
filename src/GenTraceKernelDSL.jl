module GenTraceKernelDSL

using Gen
using DynamicForwardDiff
using MacroTools

const DFD = DynamicForwardDiff

include("kernel_dsl.jl")
include("trace_token.jl")
include("inference.jl")

end # module
