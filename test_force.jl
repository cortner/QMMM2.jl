# QM/MM hybrid for right buffer region
# ForceMixing test

using JuLIP, QMMM2, SHIPs, LinearAlgebra, JuLIPMaterials
using JuLIP.MLIPs
using SKTB, Isaac, NLsolve
using SKTB: ZeroTGrand, ZeroT, MPGrid, spectrum, GammaPoint
include("./swqmmm.jl")
# include("./calculatorsnew.jl")
include("../NRL/NRLqmmm.jl")

## QM model: NRL-TB
# TB = SKTB
# NRLTB = SKTB.NRLTB
# eF = 5.5212408; # for Si
# tbm = NRLTB.NRLTBModel(:Si,
#                        ZeroTGrand(eF = eF),
#                        bzquad = TB.GammaPoint(),
#                        cutoff = :original );

r0 = rnn(:Si)
sw = StillingerWeber()
rcut = cutoff(sw)
Rmax = 12.0 * r0
ships = SWqmmm.train_ship(3, 8, wL=1.0, rinfact=0.6, rtol=5e-4)
atmax = SWqmmm.int_config(Rmax, sw)
# Nmax = 6
# atmax = NRLqmmm.vac2d_config(Nmax, tbm) 
minimise!(atmax, verbose=2, precond = FF(atmax, sw))
# add two Newton iterations to properly converge this!
H = lu( hessian(atmax) )
for i = 1:2
   g = gradient(atmax)
   x = dofs(atmax)
   x -= H \ g
   set_dofs!(atmax, x)
end
@show norm(gradient(atmax), Inf)
@show length(atmax)
@show length(dofs(atmax))
# solve the QMMM approximation
at = deepcopy(atmax)
xc = at["xcore"]
r = [ norm(x - xc) for x in positions(at) ]
Iqm = findall(r .< 6.0 * r0)
Iqmbuf = findall(r .< 9.0 * r0)
### ForceMixing
at = prepare_qmmm!(at; model = ForceMixing,
                      Vqm = sw, Vmm = ships[1], Iqm = Iqm, Iqmbuf = Iqmbuf, pbc = [false, false, false])
obj_g = x -> gradient(at, x)
# result = nlsolve(obj_g, dofs(at)) # nlsolve is too slow...
#x0, n0 = nsolimod(obj_g, dofs(at), 0) # nsolimod is not defined...
x, it_hist, ierr, x_hist = nsoli(dofs(at), obj_g)
@show size(it_hist, 1)
if ierr == 0
    set_dofs!(at, x)
    println("======== ForceMixing Done! ========")
else
    @info("Newton solver is not executed successfully!") 
end

# Plots.plot(x, y, z, color=:blue, line=:scatter, marker=:circle, xlim=[], ylim=[], zlim=[])




