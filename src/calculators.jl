
using JuLIP: Atoms, AbstractCalculator
export EnergyMixing, prepare_qmmm!,
       set_data!, set_calculator!,
       get_data


struct EnergyMixing
   QM::AbstractCalculator
   MM::AbstractCalculator
end

prepare_qmmm!(at; kwargs...) = prepare_qmmm!(at, EnergyMixing; kwargs...)

function prepare_qmmm!(at::Atoms, ::Type{EnergyMixing};
                       Vqm = nothing, Vmm = nothing,
                       Iqm = nothing)
   # perform some sanity checks
   @assert Iqm isa Vector{<: Integer}
   @assert Vqm isa AbstractCalculator
   @assert Vmm isa AbstractCalculator
   @assert sort(unique(Iqm)) == sort(Iqm)
   @assert minimum(Iqm) > 0
   @assert maximum(Iqm) <= length(at)

   # store the QM region in `at`
   set_data!(at, "qmmm_e_Iqm", Iqm)
   # store the new calculator
   set_calculator!(at, EnergyMixing(Vqm, Vmm))

   # ... and that's it already? ...
   return at
end
