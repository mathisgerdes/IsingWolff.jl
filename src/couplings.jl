#### Couplings #####
# specific couplings are for 2D lattices, but methods are general.
@enum Coupling2D K1 K2 K3 K4 K5 K6 K7 K8 K9 K10

# An array of coupling groups.
# Each group: summed over & multiply by coupling constant. 
# Each summand: lattice sites to multiply.
const CouplingSpec = AbstractVector{CPL} where {CPL<:AbstractVector{CPLT}} where {CPLT<:AbstractVector{SVector{N,Int}}} where {N}

const COUPLINGSITES = Dict(
    K1 => [  # nearest neighbor
        # each entry is a list of site indices to multiply
        # the index (0, 0) is always implicitly included
        [SVector(0, 1)],
        [SVector(1, 0)]],
    K2 => [  # diagonal nearest neighbor
        [SVector(1, 1)],
        [SVector(1, -1)]],
    K3 => [
        [SVector(0, 1), SVector(1, 0), SVector(1, 1)]],
    K4 => [
        [SVector(0, 2)],
        [SVector(2, 0)]],
    K5 => [
        [SVector(1, 2)],
        [SVector(2, 1)],
        [SVector(2, -1)],
        [SVector(-1, 2)]],
    K6 => [
        [SVector(0, 1), SVector(1, 1), SVector(1, 2)],
        [SVector(1, 0), SVector(1, 1), SVector(2, 1)],
        [SVector(1, 0), SVector(1, -1), SVector(2, -1)],
        [SVector(0, 1), SVector(1, 0), SVector(1, -1)]],
    K7 => [
        [SVector(1, -1), SVector(2, 0), SVector(1, 1)]],
    K8 => [
        [SVector(0, 1), SVector(0, 2), SVector(1, 2)],
        [SVector(0, 1), SVector(0, 2), SVector(-1, 2)],
        [SVector(1, 0), SVector(2, 0), SVector(2, -1)],
        [SVector(1, 0), SVector(2, 0), SVector(2, 1)],
        [SVector(1, 0), SVector(1, 1), SVector(1, 2)],
        [SVector(1, 0), SVector(0, 1), SVector(0, 2)]],
    K9 => [
        [SVector(2, 2)],
        [SVector(2, -2)]],
    K10 => [
        [SVector(1, 0), SVector(0, 2), SVector(1, 2)],
        [SVector(2, 0), SVector(0, 1), SVector(2, 1)]]
)


# compute hamiltonian terms; note that terms are added to existing values, set to zero before passing
function compute_couplings!(hamiltonian::AbstractVector{Int}, lattice::AbstractArray{Int, N}, couplings::CouplingSpec{N}) where {N}
    sizes = SOneTo.(size(lattice))

    for index in CartesianIndices(lattice)
        base = lattice[index]
        index = SVector{N,Int}(Tuple(index)...)

        for term_index in eachindex(hamiltonian)
            for summands in couplings[term_index]
                p = base
                for dir in summands
                    i = mod.(index + dir, sizes)
                    p *= lattice[NTuple{N,Int}(i)...]
                end
                hamiltonian[term_index] += p
            end
        end
    end
    return hamiltonian
end


# compute without pre-allocated hamiltonian array
function compute_couplings(lattice::AbstractArray, couplings::CouplingSpec)
    hamiltonian = zero(MVector{length(couplings),Int})
    compute_couplings!(hamiltonian, lattice, couplings)
    return hamiltonian
end


# contract terms with coupling constants
function compute_couplings(lattice::AbstractArray, couplingvals::AbstractVector{R}, couplings::CouplingSpec) where {R <: Real}
    hamiltonian = zero(MVector{length(couplings),Int})
    compute_couplings!(hamiltonian, lattice, couplings)
    return hamiltonian' * couplingvals
end


# contract terms with coupling constants
function compute_couplings(lattice::AbstractArray, couplingvals::AbstractVector{R}, couplings::CouplingSpec, ham::AbstractVector{Int}) where {R<:Real}
    ham .= 0
    compute_couplings!(ham, lattice, couplings)
    return ham' * couplingvals
end