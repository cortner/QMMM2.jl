
using NeighbourLists
using JuLIP: Atoms, cutoff, neighbourlist, JVecF, positions, set_positions!
using JuLIP.MLIPs: IPCollection, IPSuperBasis
using JuLIP.Potentials: site_energy, site_energy_d
import JuLIP
import SHIPs
using SHIPs: SHIPBasis, eval_basis, eval_basis_d
using LinearAlgebra: qr, norm, cond
using LowRankApprox: pqrfact

_hcat(X::AbstractVector) = hcat(X...)

_site_energy_d(args...) = hcat( site_energy_d(args...)... )

"""
l0 : site at which the site energy is evaluated
 l : site that we are perturbing
 i : direction of the perturbation (E1, E2, E3)
 h : finite-difference step
"""
function _site_energy_d2fd(basis, at::Atoms,
                           l0::Integer, l::Integer, i::Integer, h::Real)
   X = positions(at)
   X[l0] += h * evec(i)
   set_positions!(at, X)
   dBp = _site_energy_d(basis, at, l)
   X[l0] -= 2*h * evec(i)
   set_positions!(at, X)
   dBm = _site_energy_d(basis, at, l)
   X[l0] += h * evec(i)
   set_positions!(at, X)
   return (dBp - dBm) / (2*h)
end


function assemble_lsq(::Val{:dEs}, basis, config, at, h, weights)
   # import data
   w = weights["dEs"]
   dEs = config["dEs"]
   l0 = 1
   @assert length(dEs) == length(at)
   # assemble Y
   Y = zeros(length(at) * 3)
   for i = 1:length(at)
      Y[3*(i-1) .+ (1:3)] .= w * dEs[i]
   end
   # assemble A - lsq matrix
   dB = _site_energy_d(basis, at, l0)
   @assert size(dB) == (length(at), length(basis))
   A = zeros(3*length(at), length(basis))
   for iB = 1:length(basis)
      for i = 1:length(at)
         rows = 3*(i-1) .+ (1:3)
         A[rows, iB] .= w * dB[i, iB]   # dB[iB] = dEs for ith basis function
      end
   end
   return A, Y
end


function assemble_lsq(::Val{:d2Es}, basis, config, at, h, weights)
   w = weights["d2Es"]
   # import data
   d2Es = config["d2Esh"]
   nn = config["n"]   # indices of stored derivatives
   l = config["ℓ"]    # the atom where we evaluate the site energy deriv
   i = config["i"]    # the direction of the perturbation of l0 = 1
   l0 = 1
   @assert length(d2Es) == length(nn) <= length(at)
   # assemble Y
   Y = zeros(length(nn) * 3)
   for in = 1:length(nn)
      Y[3*(in-1) .+ (1:3)] .= w * d2Es[in]
   end
   # assemble A - lsq matrix
   d2B = _site_energy_d2fd(basis, at, l0, l, i, h)
   @assert size(d2B) == (length(at), length(basis))
   A = zeros(3*length(nn), length(basis))
   for iB = 1:length(basis)
      for in = 1:length(nn)
         rows = 3*(in-1) .+ (1:3)
         A[rows, iB] .= w * d2B[in, iB]
      end
   end
   return A, Y
end


function assemble_lsq(::Val{:d3Es}, basis, config, at, h, weights)
   return Matrix{Float64}(undef, 0, 0), Vector{Float64}(undef, 0)
end


function assemble_lsq(::Val{:Es}, basis, config, at, h, weights)
   w = weights["Es"]
   Y = [ w * config["Es"] ]
   A = Matrix( w * site_energy(basis, at, 1)' )
   @assert Y isa Vector
   @assert A isa Matrix
   @assert size(Y) == (1,)
   @assert size(A) == (1, length(basis))
   return A, Y
end


function get_datatype(dat::Dict)
   for k in keys(dat)
      if k == "Es"
         return Val{:Es}()
      elseif k == "dEs"
         return Val{:dEs}()
      elseif k in ["d2Es", "d2Esh"]
         return Val{:d2Es}()
      elseif k in ["d3Es", "d3Esh"]
         return Val{:d3Es}()
      end
   end
   @warn("unknown key: ")
   @show keys(dat)
   return nothing
end


function assemble_lsq(basis, D::Dict, weights::Dict)

   at = Atoms(D["at"])::Atoms
   h = D["h"]::Real
   data = D["data"]

   # --------------- assemble local matrices  ---------------
   AA = Matrix{Float64}[]
   YY = Vector{Float64}[]
   nrows = 0
   for dat in data
      # get the local lsq system
      A, Y = assemble_lsq(get_datatype(dat), basis, dat, at, h, weights)
      # store it in the AA, YY arrays
      if length(Y) != 0
         @assert size(A, 2) == length(basis)
         @assert size(A, 1) == length(Y)
         push!(AA, A)
         push!(YY, Y)
         nrows += length(Y)
      end
   end

   # --------------- assemble global matrix  ---------------
   A = zeros(nrows, length(basis))
   Y = zeros(nrows)
   irow = 0
   for (a, y) in zip(AA, YY)
      rows = irow .+ (1:size(a,1))
      A[rows, :] .= a
      Y[rows] .= y
      irow += length(rows)
   end

   return A, Y
end


function lsqfit(basis, D::Dict, weights::Dict; verbose=true, kwargs...)
   A, Y = assemble_lsq(basis, D, weights)
   qrA = pqrfact(A; rtol=1e-5, kwargs...)
   condA = cond(Matrix(qrA[:R]))
   verbose && @show condA
   c = qrA \ Y
   verbose && @show norm(c), norm(c, Inf)
   verbose && @show norm(A * c - Y) / norm(Y)
   return JuLIP.MLIPs.combine(basis, c)
end
