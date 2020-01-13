
using NeighbourLists
using JuLIP: Atoms, cutoff, neighbourlist, JVecF, positions, set_positions!
using JuLIP.MLIPs: IPCollection, IPSuperBasis
using JuLIP.Potentials: site_energy, site_energy_d
import JuLIP
import SHIPs
using SHIPs: SHIPBasis, eval_basis, eval_basis_d
using LinearAlgebra: qr, norm, cond
using LowRankApprox: pqrfact
using PrettyTables: pretty_table, ft_printf


function _asm_(djEs, djB, w, nn)
   Y = zeros(length(nn) * 3)
   A = zeros(3*length(nn), size(djB, 2))
   for in = 1:length(nn)
      rows = 3*(in-1) .+ (1:3)
      Y[3*(in-1) .+ (1:3)] .= w * djEs[in]
      for iB = 1:size(djB, 2)
         A[rows, iB] .= w * djB[in, iB]
      end
   end
   return A, Y
end

_hcat(X::AbstractVector) = hcat(X...)

_site_energy_d(args...) = hcat( site_energy_d(args...)... )

_forces(args...) = hcat( forces(args...)... )


"""
l0 : site that we are perturbing
 l : site at which the site energy is evaluated
 i : direction of the perturbation (E1, E2, E3)
 h : finite-difference step
"""
function _site_energy_d2h(basis, at::Atoms,
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

"""
l0    : first site that we are perturbing
 l    : site at which the site energy is evaluated
 k    : second site that we are perturbin
 i_l0 : direction of the perturbation of l0
 i_k  : direction of the perturbation of k
 h    : finite-difference step
"""
function _site_energy_d3h(basis, at::Atoms, l0::Integer, l::Integer,
                     k::Integer, i_l0::Integer, i_k::Integer, h::Real)
   X = positions(at)
   # -------------------------
   X[l0] += h * evec(i_l0)
   X[k] += h * evec(i_k)
   set_positions!(at, X)
   dBpp = _site_energy_d(basis, at, l)
   X[l0] -= h * evec(i_l0)
   X[k] -= h * evec(i_k)
   # -------------------------
   X[l0] += h * evec(i_l0)
   X[k] -= h * evec(i_k)
   set_positions!(at, X)
   dBpm = _site_energy_d(basis, at, l)
   X[l0] -= h * evec(i_l0)
   X[k] += h * evec(i_k)
   # -------------------------
   X[l0] -= h * evec(i_l0)
   X[k] += h * evec(i_k)
   set_positions!(at, X)
   dBmp = _site_energy_d(basis, at, l)
   X[l0] += h * evec(i_l0)
   X[k] -= h * evec(i_k)
   # -------------------------
   X[l0] -= h * evec(i_l0)
   X[k] -= h * evec(i_k)
   set_positions!(at, X)
   dBmm = _site_energy_d(basis, at, l)
   X[l0] += h * evec(i_l0)
   X[k] += h * evec(i_k)
   # -------------------------
   set_positions!(at, X)
   return (dBpp - dBpm - dBmp + dBmm) / (2*h)^2
end

"""
l0 : site that we are perturbing
 l : site at which the site energy is evaluated
 i : direction of the perturbation (E1, E2, E3)
 h : finite-difference step
"""
function _forces_dh(basis, at::Atoms, l0::Integer, i::Integer, h::Real)
    X = positions(at)
    X[l0] += h * evec(i)
    set_positions!(at, X)
    FBp = _forces(basis, at)
    X[l0] -= 2*h * evec(i)
    set_positions!(at, X)
    FBm = _forces(basis, at)
    X[l0] += h * evec(i)
    set_positions!(at, X)
    return (FBp - FBm) / (2*h)
    # return  mat( (FBp - FBm) / (2*h) )[:] |> collect
end



function assemble_lsq(::Val{:Es}, basis, config, at, w, key)
   Y = [ w * config[key] ]
   A = Matrix( w * site_energy(basis, at, 1)' )
   @assert size(Y) == (1,)
   @assert size(A) == (1, length(basis))
   return A, Y
end

function assemble_lsq(::Val{:dEs}, basis, config, at, w, key)
   # import data
   dEs = config[key]
   l0 = 1
   @assert length(dEs) == length(at)
   # assemble basis
   dB = _site_energy_d(basis, at, l0)
   @assert size(dB) == (length(at), length(basis))
   # -------------------------
   return _asm_(dEs, dB, w, 1:length(at))
end

function assemble_lsq(::Val{:d2Esh}, basis, config, at, w, key)
   # import data
   d2Es = config[key]
   l = config["l"]  # the atom where we evaluate the site energy deriv
   i = config["i"]  # the direction of the perturbation of l0 = 1
   h = config["h"]
   l0 = 1
   @assert length(d2Es) == length(at)
   # assemble basis
   d2B = _site_energy_d2h(basis, at, l0, l, i, h)
   @assert size(d2B) == (length(at), length(basis))
   # -------------------------
   return _asm_(d2Es, d2B, w, 1:length(at))
end

function assemble_lsq(::Val{:d3Esh}, basis, config, at, w, key)
   # import data
   d3Es = config[key]
   h = config["h"]
   l = config["l"]     # the atom where we evaluate the site energy deriv
   l0 = 1                   # first perturbed atom
   i_l0 = config["i_l0"]    # the direction of the perturbation of l0 = 1
   k = config["k"]          # second perturbed atom
   i_k = config["i_k"]      # direction of perturbation of k
   @assert length(d3Es) == length(at)
   # assemble basis
   d3B = _site_energy_d3h(basis, at, l0, l, k, i_l0, i_k, h)
   @assert size(d3B) == (length(at), length(basis))
   # ------------------------
   return _asm_(d3Es, d3B, w, 1:length(at))
end


function assemble_lsq(::Val{:E}, basis, config, at, w, key)
   Y = [ w * config[key] ]
   A = Matrix( w * energy(basis, at)' )
   @assert size(Y) == (1,)
   @assert size(A) == (1, length(basis))
   return A, Y
end

function assemble_lsq(::Val{:F}, basis, config, at, w, key)
   # import data
   frc = config[key]
   @assert length(frc) == length(at)
   # assemble basis
   dB = _forces(basis, at)
   @assert size(dB) == (length(at), length(basis))
   return _asm_(frc, dB, w, 1:length(at))
end

function assemble_lsq(::Val{:FC}, basis, config, at, w, key)
   # import data
   FC = config[key]
   i = config["i"]    # the direction of the perturbation of l0 = 1
   h = config["h"]
   l0 = 1             # first perturbed atom
   @assert length(FC) == length(at)
   # assemble basis
   d2B = _forces_dh(basis, at, l0, i, h)
   @assert size(d2B) == (length(at), length(basis))
   return _asm_(FC, d2B, w, 1:length(at))
end

function assemble_lsq(::Val{:EF}, basis, config, at, w, key)
   # import data
   frc = config[key]
   @assert length(frc) == length(at)
   # assemble basis
   dB = _forces(basis, at)
   @assert size(dB) == (length(at), length(basis))
   return _asm_(frc, dB, w, 1:length(at))
end



# ================== train by least square =================== #

function assemble_lsq(basis, D::Dict, weights::Dict, key="train")

   at = Atoms(D["at"])::Atoms
   data = D["data"]
   # --------------- assemble local matrices  ---------------
   AA = Matrix{Float64}[]
   YY = Vector{Float64}[]
   DTs = String[]
   nrows = 0
   for dat in data
      # get some info about the local lsq system
      dt = dat["datatype"]
      if !haskey(weights, dt); continue; end
      w = weights[dt]
      if w == 0.0; continue; end
      # assemble the local lsq system
      A, Y = assemble_lsq(Val(Symbol(dt)), basis, dat, at, w, key)
      # store it in the AA, YY arrays
      if length(Y) != 0
         @assert size(A, 2) == length(basis)
         @assert size(A, 1) == length(Y)
         push!(AA, A)
         push!(YY, Y)
         push!(DTs, dt)   # remember the data-types to assemble the errors!
         nrows += length(Y)
      end
   end
   # --------------- assemble global matrix  ---------------
   A = zeros(nrows, length(basis))
   Y = zeros(nrows)
   DT = Vector{String}(undef, nrows)
   irow = 0
   for (a, y, dt) in zip(AA, YY, DTs)
      rows = irow .+ (1:size(a,1))
      A[rows, :] .= a
      Y[rows] .= y
      DT[rows] .= dt
      irow += length(rows)
   end
   return A, Y, DT
end


function lsqfit(basis, D::Dict, weights::Dict;
                verbose=true, key="train", pqrtol = 1e-5, kwargs...)
   A, Y, DT = assemble_lsq(basis, D, weights, key)
   qrA = pqrfact(A; rtol=pqrtol, kwargs...)
   condA = cond(Matrix(qrA[:R]))
   verbose && @show condA
   c = qrA \ Y
   verbose && @show norm(c), norm(c, Inf)
   verbose && @show norm(A * c - Y) / norm(Y)
   return JuLIP.MLIPs.combine(basis, c),
          _fitinfo(basis, D, weights, kwargs, A, Y, DT, c)
end


function _fitinfo(basis, D, weights, kwargs, A, Y, DT, c)
   D = Dict("basis" => Dict(basis),
            "weights" => weights,
            "kwargs" => Dict(kwargs...),
            "rmse" => Dict{String, Any}(),
            "maxe" => Dict{String, Any}()  )
   errs = A * c - Y
   for dt in unique(DT)
      Idt = findall(DT .== dt)
      D["rmse"][dt] = norm(errs[Idt]) / norm(Y[Idt])
      D["maxe"][dt] = norm(errs[Idt], Inf) / norm(Y[Idt], Inf)
   end
   return D
end


function print_errors(fitinfo::Dict; fmt="%.3e")
   dts = ["Es", "dEs", "d2Esh", "d3Esh"]
   maxe = zeros(length(dts))
   rmse = zeros(length(dts))
   for (i, dt) in enumerate(dts)
      if haskey(fitinfo["maxe"], dt)
         maxe[i] = fitinfo["maxe"][dt]
      else
         maxe[i] = NaN
      end
      if haskey(fitinfo["rmse"], dt)
         rmse[i] = fitinfo["rmse"][dt]
      else
         rmse[i] = NaN
      end
   end
   fltfmt = ft_printf(fmt)[0]
   pretty_table( [dts maxe rmse], ["datatype", "maxe", "rmse"],
                 formatter = Dict(1 => (v,i) -> v,
                                  2 => fltfmt,
                                  3 => fltfmt) )
end
