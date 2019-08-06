
using JuLIP, QMMM2, SHIPs, PrettyTables, LinearAlgebra, Plots, JuLIPMaterials
using JuLIP.MLIPs
include(@__DIR__() * "/swqmmm.jl")

##

sw_eq = JuLIPMaterials.Si.sw_eq()

function edge_config(Rdom, calc = sw_eq, Rbuf = 2 * cutoff(calc))
   # -----------------------------------
   #  construct the configuration
   at, x0 = JuLIPMaterials.Si.edge110(:Si, Rdom + Rbuf;
                                      calc=calc, truncate=true, sym=true)
   set_data!(at, "xcore", x0)
   # -----------------------------------
   #  compute the domains
   X = positions(at)
   r = [ norm(x[1:2] - x0[1:2]) for x in X ]
   Ifree = findall(r .<= Rdom)
   set_free!(at, Ifree)
   return at, Ifree
end


##
r0 = rnn(:Si)
# domain radius of reference domain ("exact" solution)
Rmax = 75.0 * r0
# domain radius on approximation domains
RDOM = [5.0, 8.0, 12.0, 18.0, 26.0] * r0
RBUF = 2 * cutoff(sw_eq) *  ones(length(RDOM))

atmax, _ = edge_config(Rmax)
set_calculator!(atmax, sw_eq)
AT, Iinmax, Iinnext = QMMM2.Solve.cluster_sequence(atmax, RDOM, RBUF)
UU, EE, Umax, Emax  =  QMMM2.Solve.solve_sequence(atmax, AT, Iinmax, Iinnext)
err2, errinf = QMMM2.Solve.errors(atmax, AT, Iinmax, UU, Umax)
Eerr = abs.(EE .- Emax)

##
plot(RDOM, Eerr;
     xaxis = (:log, "R [Å]"),
     yaxis = (:log, "|E - E(R)| [eV]"),
     m=:o,  ms=5, lw=2, label = "|E-E[R]|",  )
plot!(RDOM[2:4], 200*RDOM[2:4].^(-3), c=:black, lw=2, ls=:dot, label = "R^-3")



##
Rmax = 60.0 * r0
atmax, _ = edge_config(Rmax)
set_calculator!(atmax, sw_eq)
f = norm.(forces(atmax))
xc = atmax["xcore"]
r = [ norm(x[1:2] - xc[1:2]) for x in positions(atmax) ]
scatter(r, f, xaxis = (:log,), yaxis = (:log,))
plot!([10.0, 100.0], 300*[10.0, 100.0].^(-3), lw=2, ls=:dash, c=:red)

X0 = positions(atmax)
minimise!(atmax; verbose=2, precond=FF(atmax))
U = positions(atmax) - X0
ee = strains(U, atmax)
scatter(r, ee.+1e-4, xaxis = (:log,), yaxis = (:log,))
plot!([10.0, 100.0], 10*[10.0, 100.0].^(-2), lw=2, ls=:dash, c=:red)

##
x, y, _ = xyz(atmax)
Ifree =  ceil.(Int, atmax.dofmgr.xfree / 3)
Iclamp = setdiff(1:length(atmax), Ifree)
scatter(x, y)
scatter(x[Ifree], y[Ifree])
scatter!(x[Iclamp], y[Iclamp])




# ##
#
# at, Ifree = edge_config(5 * r0)
# x, y, _ = xyz(at)
# Iclamp = setdiff(1:length(at), Ifree)
# scatter(x[Ifree], y[Ifree], lw=0, m=:o, ms=5, c = 1)
# scatter!(x[Iclamp], y[Iclamp], lw=0, m=:o, ms=5, c = 2)
#
# ##
# set_calculator!(at, sw_eq)
# minimise!(at, verbose=2, precond = FF(at, sw_eq))
# x, y, _ = xyz(at)
# scatter(x[Ifree], y[Ifree], lw=0, m=:o, ms=5, c = 1)
# scatter!(x[Iclamp], y[Iclamp], lw=0, m=:o, ms=5, c = 2)
#
# ##
#
# # H = hessian(at)
# # x = dofs(at)
# # g = gradient(at)
# # @show norm(g, Inf)
# # x -= H \ g
# # g = gradient(at, x)
# # @show norm(g, Inf)
# # x -= H \ g
# # g = gradient(at, x)
# # @show norm(g, Inf)
#
# ##
# n = 3
# at = AT[n]
# x, y, _ = xyz(at)
# Ifree =  unique(ceil.(Int, at.dofmgr.xfree / 3))
# Iclamp = setdiff(1:length(at), Ifree)
# scatter(x[Ifree], y[Ifree], m=:o, ms=5, c = 1)
# scatter!(x[Iclamp], y[Iclamp], m=:o, ms=5, c = 2)
#
#
#
# length(at) == length(AT[1])
# sort(at.X) ≈ sort(AT[1].X)
# X1 = sort(positions(at))
# X2 = sort(positions(AT[1]))
# X1 = [x - X1[1] for x in X1]
# X2 = [x - X2[1] for x in X2]
# mat(X1)
# mat(X2)
# cell(at)
#
#
#
#
# cell(AT[1])
#
#
#
#
# cell(atmax)
