# QM/MM hybrid approximation

# using QMMM2
using PrettyTables, LinearAlgebra, Plots, DataFrames
using JuLIP, SHIPs, JuLIPMaterials
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
train_size = 1;
at_train = bulk(:Si, cubic=true) * (train_size, train_size, 1)
rcut = [tbm_cutoff_d1, tbm_cutoff_d2, tbm_cutoff_d3]
at = at_train
D = QMMM2.data_djEs_sketch(at, 1e-3, rcut)
QMMM2.eval_dataset_tb!(D, tbm; key="NRLTB")

## train the MM model 🚢
wL = 1.7          # 1.0 1.5  1.75
rin = 0.7         # 0.6 0.7 0.8
rtol = 1e-6       # 1e-5, 1e-10, 1e-15 ...
bo = 2            # 2, 3, 4, 5
deg = 8          # 5, 10, 12, 15

# TODO： careful whether include d3Es, also the functions in NRLqmmm.jl ...
# weights = Dict("Es" => 10.0, "dEs" => 10.0, "d2Esh" => 1.0)
weights = Dict("Es" => 10.0, "dEs" => 1.0, "d2Esh" => 1.0, "d3Esh" => 1.0)

basis = NRLqmmm.get_basis(bo, deg; rinfact=rin, wL=wL)
@show length(basis)
# TODO: test the parameters
🚢, fitinfo = NRLqmmm.train_ship(D, bo, deg,
               wL = wL, rinfact = rin, weights = weights, rtol = rtol)
@show fitinfo["rmse"]


# Solve the reference problem, domain radius of reference ("exact" solution)
Nmax = 8
# Rmax = Nmax * r0
atmax = NRLqmmm.vac2d_config(Nmax, tbm);
atmax0 = deepcopy(atmax);
X0 = positions(atmax0);
E0_max = energy(atmax0);
# TODO: use SW to relax the lattice to obtain an initial configuration
# sw = StillingerWeber()
# set_calculator!(atmax, sw)
# minimise!(atmax, precond = FF(atmax, sw))
# use NRL-TB
println("number of atoms = ", length(atmax), " start relaxation for pure QM...")
set_calculator!(atmax, tbm);
optresult = minimise!(atmax; verbose = 2, gtol = 1.0e-3);
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
E_max = E0_max - energy(atmax)


# plot fuction
function plot(at::Atoms{T}; Iqm = 1:length(at), i = 1) where T <: Number
   x, y, _ = xyz(at)
   figure(i)
   PyPlot.plot(x, y, "bo", x[Iqm], y[Iqm], "ro", markersize = 5)
   axis("equal")
end
# relaxation and plot the configuration within a QMMM decoposition
α = 0.03
RQM = 10.0
at = deepcopy(atmax0);  # at = deepcopy(atmax)
xc = at["xcore"];
r = [ norm(x - xc) for x in positions(at) ];
Iqm = findall(r .< RQM);
# QMMM relaxation
at = prepare_qmmm!(at, EnergyMixing; Vqm = tbm, Vmm = 🚢, Iqm = Iqm);
# start minimization
for k = 1 : 20
   frc = forces(at);
   E = energy(at);
   gnorm = norm(frc, Inf);  # gnorm = norm(frc, 2);
   X0 = positions(at);
   set_positions!(at, X0.+α*frc);
   println("energy = ", E, "  gradient norm = ", gnorm)
   plot(at, Iqm = Iqm, i = k)
   # pause(5.0) close()
   if gnorm < 1.0e-3
      break
   end
end


# Setup sequence of QM-regions and MM potentials
NQM = [6, 5, 4, 3]
RQM = NQM * r0
errE = zeros(Float64, size(NQM))
err2 = zeros(Float64, size(NQM))
err∞ = zeros(Float64, size(NQM))
errat = []
for n = 1 : length(NQM)
   @show RQM[n]
   at = deepcopy(atmax0)  # at = deepcopy(atmax)
   xc = at["xcore"]
   r = [ norm(x - xc) for x in positions(at) ]
   Iqm = findall(r .< RQM[n])
   # QMMM relaxation
   # set_free!(at, findall(r .<=  RQM[n]))
   at = prepare_qmmm!(at, EnergyMixing; Vqm = tbm, Vmm = 🚢, Iqm = Iqm)
   E0 = energy(at)
   at0 = deepcopy(at)
   optresult = minimise!(at0; verbose = 2, gtol = 1.0e-3, g_calls_limit = 20)
   E = E0 - energy(at0)
   # error of the energy and configuration
   errE[n] = E - E_max
   U = positions(at) - positions(atmax)
   push!(errat, U)
   # TODO: need the implementation of cutoff(tbm)
   ee = JuLIPMaterials.strains(U, atmax0; rcut = 1.3 * rnn(:Si))
   # err2[n] = norm(ee, 2)
   # err∞[n] = norm(ee, Inf)
end
@show errE
@show err2
@show err∞




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
