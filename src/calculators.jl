
using JuLIP: Atoms, AbstractCalculator
export EnergyMixing, prepare_qmmm!

import JuLIP: energy, forces



# ------------------------------------------------------------------

function get_domains(at::Atoms)
   Iqm = get_data(at, "qmmm_e_Iqm")
   # Iqmbuf = get_data(at, "qmmm_e_Iqmbuf")
   Imm = setdiff(1:length(at), Iqm)
   return Iqm, Imm, nothing
end

struct EnergyMixing{T} <: AbstractCalculator
   atqm::Atoms{T}
   QM::AbstractCalculator
   MM::AbstractCalculator
end

function energy(calc::EnergyMixing{T}, at::Atoms{T}) where {T}
   Iqm, Imm, Iqmbuf = get_domains(at)
   # set_positions!(calc.atqm, positions(at)[Iqmbuf])
   Eqm = energy(calc.QM, at; domain = Iqm)
   Emm = energy(calc.MM, at; domain = Imm)
   return Eqm + Emm
end

function forces(calc::EnergyMixing{T}, at::Atoms{T}) where {T}
   Iqm, Imm, Iqmbuf = get_domains(at)
   # set_positions!(calc.atqm, positions(at)[Iqmbuf])
   Fqm = forces(calc.QM, at; domain = Iqm)
   Fmm = forces(calc.MM, at; domain = Imm)
   return Fqm + Fmm
end

# ------------------------------------------------------------------


struct ForceMixing{T}
   atqm::Atoms{T}
   QM::AbstractCalculator
   MM::AbstractCalculator
end

function forces(calc::ForceMixing{T}, at::Atoms{T}) where {T}
   Iqm, Imm, Iqmbuf = get_domains(at)
   # TODO: the following forces are not correctly implemented
   Fqm = forces(calc.QM, at; domain = Iqm)
   Fmm = forces(calc.MM, at; domain = Imm)
   return Fqm + Fmm
end


# ------------------------------------------------------------------


prepare_qmmm!(at; kwargs...) = prepare_qmmm!(at, EnergyMixing; kwargs...)

function prepare_qmmm!(at::Atoms, ::Type{EnergyMixing};
                       Vqm = nothing, Vmm = nothing,
                       Iqm = nothing)
   # perform some sanity checks
   Iqm = collect(Iqm)
   @assert Iqm isa Vector{<: Integer}
   @assert Vqm isa AbstractCalculator
   @assert Vmm isa AbstractCalculator
   @assert sort(unique(Iqm)) == sort(Iqm)
   @assert minimum(Iqm) > 0
   @assert maximum(Iqm) <= length(at)

   # create an atoms object for the QM calculator
   # => TODO: add buffer, carve out region, etc...

   # store the QM region in `at`
   at["qmmm_e_Iqm"] = Iqm
   # store the new calculator
   set_calculator!(at, EnergyMixing(at, Vqm, Vmm))

   return at
end
