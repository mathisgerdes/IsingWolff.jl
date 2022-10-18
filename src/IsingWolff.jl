__precompile__()

module IsingWolff

using StaticArrays
using Random

const βc = log(1 + √2)/2
const beta_crit = βc


include("wolff.jl")  # basic Wolff algorithm
include("couplings.jl")  # compute hamiltonian terms given couplings
include("wolffplus.jl")  # Wolff + Metropolis Hastings for perturbed couplings


end  # IsingWolff
