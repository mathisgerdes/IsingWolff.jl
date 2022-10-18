#### Wolff algorithm + Metropolis Hastings step #####
function wolffstep!(
    rng, lattice::LatticeArray{N}, energy_prev, couplingvals, couplings::CouplingSpec{N}, p_add,
    # precomputed / preallocated resources
    lattice_indices, directions, stack, ham) where {N}

    wolffstep!(rng, lattice, p_add, directions, lattice_indices, stack)
    energy_new = compute_couplings(lattice, couplingvals, couplings, ham)

    accept_prob = exp(energy_new - energy_prev)  # assume plus sign in exp(H)
    return energy_new, rand(rng) < accept_prob  # true if accept
end


function wolffsample!(
    samples::AbstractArray{T},
    rng::AbstractRNG,
    lattice::AbstractArray{Int,N},
    β::Real,
    couplingvals::AbstractVector{R},
    couplings::CouplingSpec{N},
    nsamples,
    keep_every,
    ntherm
) where {T<:Union{Int,Bool},N,R<:Real}
    lattice = LatticeArray{N,Tuple{size(lattice)...}}(lattice)
    p_add = 1 - exp(-2β)
    lat_ind = CartesianIndices(lattice)

    # can go up and down in each of the N dimensions
    directions = SVector{2N}([
        dir * SVector{N}(setindex!(zeros(Int64, N), 1, i))
        for i in 1:N for dir in [1, -1]])

    # stack is allocated once then reused in the wolff step
    stack = zero(MVector{length(lattice),SVector{N,Int64}})
    # hamiltonian term vector, to be reused in wolff step
    ham = zero(MVector{length(couplingvals),Int})

    lattice_prev = copy(lattice)
    energy = energy_prev = compute_couplings(lattice, couplingvals, couplings, ham)

    # thermalization
    for _ in 1:ntherm
        energy, accept = wolffstep!(rng, lattice, energy_prev, couplingvals, couplings, p_add, lat_ind, directions, stack, ham)
        if accept
            copy!(lattice_prev, lattice)
            energy_prev = energy
        end
    end

    savesample!(1, samples, lattice_prev, lat_ind)

    naccept = 0
    for i in 2:nsamples
        for _ in 1:keep_every
            energy, accept = wolffstep!(rng, lattice, energy_prev, couplingvals, couplings, p_add, lat_ind, directions, stack, ham)
            if accept
                naccept += 1
                copy!(lattice_prev, lattice)
                energy_prev = energy
            end
        end
        savesample!(i, samples, lattice_prev, lat_ind)
    end

    return samples, naccept / (keep_every * (nsamples - 1))
end


# couplings as list of Coupling2D
function wolffsample!(samples::AbstractArray, rng::AbstractRNG, lattice::AbstractArray{Int}, β, couplingvals, couplings::AbstractArray{Coupling2D}, nsamples, keep_every, ntherm)
    couple_spec = [COUPLINGSITES[c] for c in couplings]
    return wolffsample!(samples, rng, lattice, β, couplingvals, couple_spec, nsamples, keep_every, ntherm)
end


# start with Boolean lattice
function wolffsample!(samples::AbstractArray, rng::AbstractRNG, lattice::AbstractArray{Bool}, β, couplingvals, couplings, nsamples, keep_every, ntherm)
    return wolffsample!(samples, rng, (s -> s ? 1 : -1).(lattice), β, couplingvals, couplings, nsamples, keep_every, ntherm)
end


# with random initial lattice
function wolffsample!(samples::AbstractArray, rng::AbstractRNG, lattice_shape::Tuple, β, couplingvals, couplings, nsamples, keep_every, ntherm)
    lattice = rand(rng, (-1, 1), lattice_shape)
    return wolffsample!(samples, rng, lattice, β, couplingvals, couplings, nsamples, keep_every, ntherm)
end


# with random seed
function wolffsample!(samples::AbstractArray, seed::Int, lattice::Union{Tuple,AbstractArray}, β, couplingvals, couplings, nsamples, keep_every, ntherm)
    rng = Xoshiro(seed)
    return wolffsample!(samples, rng, lattice, β, couplingvals, couplings, nsamples, keep_every, ntherm)
end


# with default rng
function wolffsample!(samples::AbstractArray, lattice::Union{Tuple,AbstractArray}, β, couplingvals, couplings, nsamples, keep_every, ntherm)
    rng = Random.default_rng()
    return wolffsample!(samples, rng, lattice, β, couplingvals, couplings, nsamples, keep_every, ntherm)
end


# without pre-allocated sample array (shape)
function wolffsample(rng::AbstractRNG, lattice_shape::NTuple{N,Int}, β, couplingvals, couplings, nsamples, keep_every, ntherm) where {N}
    samples = Array{Int,N + 1}(undef, nsamples, lattice_shape...)
    return wolffsample!(samples, rng, lattice_shape, β, couplingvals, couplings, nsamples, keep_every, ntherm)
end


# without pre-allocated sample array (array)
function wolffsample(rng::AbstractRNG, lattice::AbstractArray{T,N}, β, couplingvals, couplings, nsamples, keep_every, ntherm) where {T,N}
    samples = Array{Int,N + 1}(undef, nsamples, size(lattice)...)
    return wolffsample!(samples, rng, lattice, β, couplingvals, couplings, nsamples, keep_every, ntherm)
end

# without pre-allocated sample array & without rng
function wolffsample(lattice::Union{AbstractArray,Tuple}, β, couplingvals, couplings, nsamples, keep_every, ntherm)
    rng = Random.default_rng()
    return wolffsample(rng, lattice, β, couplingvals, couplings, nsamples, keep_every, ntherm)
end
