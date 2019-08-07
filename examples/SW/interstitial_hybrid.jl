
using JuLIP, QMMM2, SHIPs, PrettyTables, LinearAlgebra, Plots,
      DataFrames, JuLIPMaterials
using JuLIP.MLIPs
include(@__DIR__() * "/swqmmm.jl")


## reference model and MM potential(s) (cf. mmpot_test.jl)
r0 = rnn(:Si)
sw = StillingerWeber()
ships = [ SWqmmm.train_ship(3, deg, wL=1.0, rinfact=0.6, rtol=5e-4)
          for deg in [3,4,5] ]
# all ships have the same cutoff as sw:
rcut = cutoff(sw)

## Solve the reference problem
# domain radius of reference domain ("exact" solution)
Rmax = 16.0 * r0
atmax0 = SWqmmm.int_config(Rmax, sw)
atmax  = SWqmmm.int_config(Rmax, sw)
X0 = positions(atmax0)
E0_max =  energy(atmax0)
minimise!(atmax, precond = FF(atmax, sw))
# add two Newton iterations to properly converge this!
H = lu( hessian(atmax) )
for i = 1:2
   g = gradient(atmax)
   x = dofs(atmax)
   x -= H \ g
   set_dofs!(atmax, x)
end
@show norm(gradient(atmax), Inf)
E1_max = energy(atmax)
dE_max = E0_max - E1_max


## Setup sequence of QM-regions and MM potentials
RQM = [2.0, 3.0, 5.0, 7.0] * r0

errE = zeros(length(RQM), length(ships))
err2 = zeros(length(RQM), length(ships))
errinf = zeros(length(RQM), length(ships))

for n = length(RQM):-1:1, imm = length(ships):-1:1
   @show RQM[n], imm
   at = deepcopy(atmax)
   xc = at["xcore"]
   r = [ norm(x - xc) for x in positions(at) ]
   Iqm = findall(r .< RQM[n])
   at = prepare_qmmm!(at, EnergyMixing;
                      Vqm = sw, Vmm = ships[imm][1], Iqm = Iqm)
   at0 = deepcopy(at)
   set_positions!(at0, X0)
   E0 = energy(at0)
   success = false
   try
      optresult = minimise!(at; precond = FF(at, sw), verbose=2,
                                g_calls_limit=20)
      if optresult.g_residual < 1e-3
         success = true
      end
   catch
      @info("`minimise!` crashed!")
   end
   if !success
      errE[n, imm] = NaN
      err2[n, imm] = NaN
      errinf[n, imm] = NaN
      @info("Optimisation did not converge")
   else
      @info("Optimisation converged; improve with Newton step")
      x = dofs(at) - H \ gradient(at)
      set_dofs!(at, x)
      @show norm(gradient(at), Inf)
      @info("Compute errors")
      E1 = energy(at)
      dE = E0 - E1
      errE[n, imm] = abs(dE - dE_max)

      U = positions(at) - positions(atmax)
      ee = JuLIPMaterials.strains(U, atmax0)
      err2[n, imm] =  norm(ee, 2)
      errinf[n, imm] = norm(ee, Inf)
      @show abs(dE - dE_max)
      @show norm(ee, 2)
   end
end


##

using Printf
println("-"^85)
println("       |" * prod("|         Model$n          " for n = 1:3))
println("  Rqm  |" * "|  E [eV] 2-norm  ∞-norm  "^3)
println("-"^85)
for n =  1:length(RQM)
   @printf(" %5.1f |", RQM[n])
   for imm = 1:length(ships)
      if isnan(errE[n, imm])
         print("|" * "    x   "^3 * " ")
      else
         @printf("| %3.1e %3.1e %3.1e ",
                 errE[n, imm], err2[n, imm], errinf[n, imm])
      end
   end
   println()
end



##
#
# n = 1; imm = 1
# @show RQM[n], imm
# at = deepcopy(atmax)
# xc = at["xcore"]
# r = [ norm(x - xc) for x in positions(at) ]
# Iqm = findall(r .< RQM[n])
# at = prepare_qmmm!(at, EnergyMixing;
#                    Vqm = sw, Vmm = ships[imm][1], Iqm = Iqm)
# at0 = deepcopy(at)
# set_positions!(at0, X0)
# E0 = energy(at0)
# optresult = minimise!(at, precond = FF(at, sw), verbose=2, g_calls_limit=30)
# if optresult.g_residual > 1e-3
#
# E1 = energy(at)
# dE = E0 - E1
# errE[n, imm] = abs(dE - dE_max)
#
# U = positions(at) - positions(atmax)
# ee = JuLIPMaterials.strains(U, atmax0)
# err2[n, imm] =  norm(ee, 2)
# errinf[n, imm] = norm(ee, Inf)
# @show norm(ee, 2)
# @show abs(dE - dE_max)
