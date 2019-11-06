# QM/MM hybrid for right buffer region
# EnergyMixing test

using JuLIP, QMMM2, SHIPs, LinearAlgebra, JuLIPMaterials
using JuLIP.MLIPs
using SKTB, Isaac, NLsolve
include("./swqmmm.jl")
# include("./calculatorsnew.jl")

r0 = rnn(:Si)
sw = StillingerWeber()
ships = SWqmmm.train_ship(3, 8, wL=1.0, rinfact=0.6, rtol=5e-4)
# all ships have the same cutoff as sw:
rcut = cutoff(sw)

### EnergyMixing
## Solve the reference problem
# domain radius of reference domain ("exact" solution)
Rmax = 20.0 * r0
atmax = SWqmmm.int_config(Rmax, sw)   # interstitial
atmax0 = deepcopy(atmax)
# atmax = SWqmmm.vac_config(Rmax, sw)   # vacancy
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

at = deepcopy(atmax)
xc = at["xcore"]
r = [ norm(x - xc) for x in positions(at) ]
Iqm = findall(r .< 6.0 * r0)
Iqmbuf = findall(r .< 9.0 * r0)
at = prepare_qmmm!(at; model = EnergyMixing,
                Vqm = sw, Vmm = ships[1], Iqm = Iqm, Iqmbuf = Iqmbuf, pbc = [false, false, false])
optresult = minimise!(at; precond = FF(at, sw), verbose = 2,
                        g_calls_limit = 20)
# JuLIP.Testing.fdtest(at.calc, at)





