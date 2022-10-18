#### Basic Wolff algorithm #####
const LatticeArray = SizedArray{S,Int,N,N,TData} where {TData<:AbstractArray{Int,N}} where {N,S}

function wolffstep!(
    rng, lattice::LatticeArray{N}, p_add,
    # precomputed / preallocated resources
    directions, lattice_indices, stack) where {N}

    start = rand(rng, lattice_indices)
    stack[1] = SVector(Tuple(start)...)
    shape = SVector{N}(size(lattice)...)
    stack_index = 1

    clustersign = @inbounds lattice[start]
    newsign = -clustersign
    @inbounds lattice[start] = newsign

    while (stack_index > 0)
        prev = @inbounds stack[stack_index]
        stack_index -= 1

        for direction in directions
            next = @. mod(prev + direction - 1, shape) + 1
            next_ci = CartesianIndex{N}(next...)

            if @inbounds(lattice[next_ci]) == clustersign
                if rand(rng) < p_add
                    stack_index += 1
                    @inbounds stack[stack_index] = next
                    @inbounds lattice[next_ci] = newsign
                end
            end
        end
    end
    nothing
end

function savesample!(i, samples::AbstractArray{Bool}, lattice, lattice_indices)
    @. samples[i, lattice_indices] = (lattice == 1)
end

function savesample!(i, samples::AbstractArray{Int}, lattice, lattice_indices)
    @. samples[i, lattice_indices] = lattice
end

function wolffsample!(
    samples::AbstractArray{T},
    rng::AbstractRNG,
    lattice::AbstractArray{Int,N},
    β::Real,
    nsamples,
    keep_every=100,
    ntherm=0
) where {N,T<:Union{Int,Bool}}
    lattice = LatticeArray{N,Tuple{size(lattice)...}}(lattice)
    p_add = 1 - exp(-2β)
    lat_ind = CartesianIndices(lattice)

    # can go up and down in each of the N dimensions
    directions = SVector{2N}([
        dir * SVector{N}(setindex!(zeros(Int64, N), 1, i))
        for i in 1:N for dir in [1, -1]])

    # stack is allocated once then reused in the wolff step
    stack = Vector{SVector{N,Int64}}(undef, length(lattice))

    # thermalization
    for _ in 1:ntherm
        wolffstep!(rng, lattice, p_add, directions, lat_ind, stack)
    end

    savesample!(1, samples, lattice, lat_ind)

    for i in 2:nsamples
        for _ in 1:keep_every
            wolffstep!(rng, lattice, p_add, directions, lat_ind, stack)
        end
        savesample!(i, samples, lattice, lat_ind)
    end

    return samples
end


# start with Boolean lattice
function wolffsample!(samples::AbstractArray, rng::AbstractRNG, lattice::AbstractArray{Bool}, β, nsamples, keep_every, ntherm)
    return wolffsample!(samples, rng, (s -> s ? 1 : -1).(lattice), β, nsamples, keep_every, ntherm)
end

# start with random lattice
function wolffsample!(samples::AbstractArray, rng::AbstractRNG, lattice_shape::Tuple, β, nsamples, keep_every, ntherm)
    lattice = rand(rng, (-1, 1), lattice_shape)
    return wolffsample!(samples, rng, lattice, β, nsamples, keep_every, ntherm)
end


# with random seed
function wolffsample!(samples::AbstractArray, seed::Int, lattice::Union{Tuple,AbstractArray}, β, nsamples, keep_every, ntherm)
    rng = Xoshiro(seed)
    return wolffsample!(samples, rng, lattice, β, nsamples, keep_every, ntherm)
end


# with default rng
function wolffsample!(samples::AbstractArray, lattice::Union{Tuple,AbstractArray}, β, nsamples, keep_every, ntherm)
    rng = Random.default_rng()
    return wolffsample!(samples, rng, lattice, β, nsamples, keep_every, ntherm)
end


# without pre-allocated sample array (shape)
function wolffsample(rng::Union{Int,AbstractRNG}, lattice_shape::NTuple{N,Int}, β, nsamples, keep_every, ntherm) where {N}
    samples = Array{Int,N + 1}(undef, nsamples, lattice_shape...)
    wolffsample!(samples, rng, lattice_shape, β, nsamples, keep_every, ntherm)
    return samples
end


# without pre-allocated sample array (array)
function wolffsample(rng::Union{Int,AbstractRNG}, lattice::AbstractArray{T,N}, β, nsamples, keep_every, ntherm) where {T,N}
    samples = Array{Int,N + 1}(undef, nsamples, size(lattice)...)
    wolffsample!(samples, rng, lattice, β, nsamples, keep_every, ntherm)
    return samples
end


# without pre-allocated sample array & without rng
function wolffsample(lattice::Union{Tuple,AbstractArray}, β, nsamples, keep_every, ntherm) where {N}
    rng = Random.default_rng()
    return wolffsample(rng, lattice, β, nsamples, keep_every, ntherm)
end