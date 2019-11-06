
# energy mixing (buf) + force mixing (buf)

using JuLIP: Atoms, AbstractCalculator, JVec, positions, cell, get_data
using JuLIP: set_cell!, set_free!, set_pbc!, set_calculator!, set_positions!, chemical_symbol
using NearestNeighbors
export EnergyMixing, ForceMixing, prepare_qmmm!

import JuLIP: energy, forces

# ------------------------------------------------------------------

function get_domains(at::Atoms)
   Iqm = get_data(at, "qmmm_e_Iqm")
   Iqmbuf = get_data(at, "qmmm_e_Iqmbuf")
   Imm = setdiff(1:length(at), Iqm)
   return Iqm, Imm, Iqmbuf
end

struct EnergyMixing{T} <: AbstractCalculator
   atqmbuf::Atoms{T} 
   QM::AbstractCalculator
   MM::AbstractCalculator
end

function _update_atqm!(atqm, at)
   Iqm, Imm, Iqmbuf = get_domains(at)
   idxqm_in_atqmbuf = get_data(at, "idxqm_in_atqmbuf")
   set_positions!(atqm, positions(at)[Iqmbuf])
   set_cell!(atqm, cell(at)) # QMMM2.Solve.autocell12(at, atqm)) if pbc is used...
   return idxqm_in_atqmbuf, Iqm, Imm, Iqmbuf
end

function energy(calc::EnergyMixing{T}, at::Atoms{T}) where {T}
   idxqm_in_atqmbuf, _, Imm, _ = _update_atqm!(calc.atqmbuf, at)
   Eqm = energy(calc.QM, calc.atqmbuf; domain = idxqm_in_atqmbuf)
   Emm = energy(calc.MM, at; domain = Imm)
   return Eqm + Emm
end

function forces(calc::EnergyMixing{T}, at::Atoms{T}) where {T}
   idxqm_in_atqmbuf, _, Imm, Iqmbuf = _update_atqm!(calc.atqmbuf, at)
   Fqm = forces(calc.QM, calc.atqmbuf; domain = idxqm_in_atqmbuf)
   Fmm = forces(calc.MM, at; domain = Imm)
   Fh = copy(Fmm)
   Fh[Iqmbuf] .+= Fqm 
   return Fh
end

# ------------------------------------------------------------------

struct ForceMixing{T} <: AbstractCalculator
   atqmbuf::Atoms{T}
   QM::AbstractCalculator
   MM::AbstractCalculator
end

function forces(calc::ForceMixing{T}, at::Atoms{T}) where {T}
   idxqm_in_atqmbuf, Iqm, Imm, _ = _update_atqm!(calc.atqmbuf, at)
   F = forces(calc.MM, at)
   F[Iqm] = forces(calc.QM, calc.atqmbuf)[idxqm_in_atqmbuf]
   return F
end

# ------------------------------------------------------------------
# only for energy mixing right now 
# should change it to force mixing
prepare_qmmm!(at; model = EnergyMixing, kwargs...) = prepare_qmmm!(at, model; kwargs...)

function prepare_qmmm!(at::Atoms, model::Union{Type{EnergyMixing}, Type{ForceMixing}};
                       Vqm = nothing, Vmm = nothing,
                       Iqm = nothing, Iqmbuf = nothing, pbc = [false, false, true])
   # perform some sanity checks
   Iqm = collect(Iqm)
   Iqmbuf = collect(Iqmbuf)
   @assert Iqm isa Vector{<: Integer}
   @assert Vqm isa AbstractCalculator
   @assert Vmm isa AbstractCalculator
   @assert sort(unique(Iqm)) == sort(Iqm)
   @assert sort(unique(Iqmbuf)) == sort(Iqmbuf)
   @assert minimum(Iqm) > 0
   @assert maximum(Iqm) <= length(at)

   # store the QM region in `at`
   idxqm_in_atqmbuf, _ = findin(positions(at)[Iqm], positions(at)[Iqmbuf])
   at["qmmm_e_Iqm"] = Iqm
   at["qmmm_e_Iqmbuf"] = Iqmbuf
   at["idxqm_in_atqmbuf"] = idxqm_in_atqmbuf

   # create a QM buffer region
   # construct Iqm_buf
   atqmbuf = Atoms(chemical_symbol(at.Z[1]), positions(at)[Iqmbuf])
   set_free!(atqmbuf, idxqm_in_atqmbuf)
   set_pbc!(atqmbuf, pbc)

   # store the new calculator
   if model == EnergyMixing
      set_calculator!(at, EnergyMixing(atqmbuf, Vqm, Vmm))
   else 
      set_calculator!(at, ForceMixing(atqmbuf, Vqm, Vmm))
   end

   return at
end

# --------------------- should be put somewhere --------------------

"""
`findin(Xsm, Xlge)`
from JuLIPMaterials/src/commom.jl
"""
function findin(Xsm::Vector{JVec{T}}, Xlge::Vector{JVec{T}}) where T <: AbstractFloat
   # find the nearest neighbours of Xsm points in Xlge
   tree = KDTree(Xlge)
   # construct the Xsm -> Xlge mapping
   Ism = zeros(Int, length(Xsm))
   Ilge = zeros(Int, length(Xlge))
   for (n, x) in enumerate(Xsm)
      i = inrange(tree, Xsm[n], sqrt(eps(T)))
      # println(length(i))
      if isempty(i)
         Ism[n] = 0         # - Ism[i] == 0   if  Xsm[i] ∉ Xlge
      elseif length(i) > 1
         error("`inrange` found two neighbours")
      else
         Ism[n] = i[1]      # - Ism[i] == j   if  Xsm[i] == Xlge[j]
         Ilge[i[1]] = n     # - Ilge[j] == i  if  Xsm[i] == Xlge[j]
      end
   end
   # - if Xlge[j] ∉ Xsm then Ilge[j] == 0
   return Ism, Ilge
end
