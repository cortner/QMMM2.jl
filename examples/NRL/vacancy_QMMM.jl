# QM/MM hybrid approximation

using JuLIP, QMMM2, SHIPs, PrettyTables, LinearAlgebra, Plots,
      DataFrames, JuLIPMaterials
using JuLIP.MLIPs
using SKTB
using SKTB: ZeroTGrand, ZeroT, MPGrid, spectrum, GammaPoint
include(@__DIR__() * "/NRLqmmm.jl")

TB = SKTB
NRLTB = SKTB.NRLTB


## QM model: NRL-TB
eF = 5.5212408; # for Si
tbm = NRLTB.NRLTBModel(:Si,
                       ZeroTGrand(eF = eF),
                       bzquad = TB.GammaPoint(),
                       cutoff = :original );
r0 = rnn(:Si)
# QM cutoff: from the locality
tbm_cutoff_d1 = 5.0 * r0
tbm_cutoff_d2 = 4.0 * r0
tbm_cutoff_d3 = 3.0 * r0

## create the database from the QM (NRL-TB) model
train_size = 5;
at_train = bulk(:Si, cubic=true) * (train_size, train_size, 1)
rcut = [tbm_cutoff_d1, tbm_cutoff_d2, tbm_cutoff_d3]
at = at_train
D = QMMM2.data_djEs_sketch(at, 1e-3, rcut)
QMMM2.eval_dataset_tb!(D, tbm; key="NRLTB")

## train the MM model 🚢
wL = 1.7          # 1.0 1.5  1.75
rin = 0.7         # 0.6 0.7 0.8
rtol = 1e-10      # 1e-5, 1e-4, ...
bo = 3            # 4, 5, 6
deg = 15          # 5, 10, 12, 15
weights = Dict("Es" => 10.0, "dEs" => 10.0, "d2Esh" => 1.0)
# TODO: test the parameters
🚢, fitinfo = NRLqmmm.train_ship(D, bo, deg,
               wL=wL, rinfact=rin, weights=weights, rtol=rtol)
@show fitinfo["rmse"]


# Solve the reference problem, domain radius of reference ("exact" solution)
Nmax = 8
# Rmax = Nmax * r0
atmax = NRLqmmm.vac2d_config(Nmax, tbm)
atmax0 = deepcopy(atmax)
X0 = positions(atmax0)
E0_max = energy(atmax0)
# TODO: use SW to relax the lattice to obtain an initial configuration
# sw = StillingerWeber()
# set_calculator!(atmax, sw)
# minimise!(atmax, precond = FF(atmax, sw))
# use NRL-TB
set_calculator!(atmax, tbm)
minimise!(atmax)
E0 = energy(atmax)
# add two Newton iterations to properly converge this!
# TODO: fix the following newton iteration for tbm
# H = lu( hessian(atmax) )
# for i = 1:2
#    g = gradient(atmax)
#    x = dofs(atmax)
#    x -= H \ g
#    set_dofs!(atmax, x)
# end
@show norm(gradient(atmax), Inf)
E1_max = energy(atmax)
dE_max = E0_max - E1_max


## Setup sequence of QM-regions and MM potentials
NQM = [3.5, 4, 4.5, 5]
RQM = NQM * r0
E = zeros(size(NQM))
for n = 1 : length(NQM)
   @show RQM[n]
   at = deepcopy(atmax)
   xc = at["xcore"]
   r = [ norm(x - xc) for x in positions(at) ]
   Iqm = findall(r .< RQM[n])

   at = prepare_qmmm!(at, EnergyMixing; Vqm = tbm, Vmm = 🚢, Iqm = Iqm)
   at0 = deepcopy(at)
   optresult = minimise!(at0; verbose = 2, g_calls_limit = 20)
   E[n] = energy(at0)
end
err = abs.(E .- E0)
@show err




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
                      Vqm = tbm, Vmm = ships[imm][1], Iqm = Iqm)
   at0 = deepcopy(at)
   set_positions!(at0, X0)
   E0 = energy(at0)
   success = false
   try
      optresult = minimise!(at; # precond = FF(at, sw),
                                verbose = 2, g_calls_limit = 20)
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
      @info("Optimisation converged; improve with Newton steps")
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

## output
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
