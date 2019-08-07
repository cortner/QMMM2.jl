
using JuLIP, QMMM2, SHIPs, PrettyTables, LinearAlgebra, Plots, JuLIPMaterials
using JuLIP.MLIPs
include(@__DIR__() * "/swqmmm.jl")

sw_eq = SWqmmm.sw_eq

##
r0 = rnn(:Si)
# domain radius of reference domain ("exact" solution)
Rmax = 75.0 * r0
# domain radius on approximation domains
RDOM = [5.0, 8.0, 12.0, 18.0, 26.0] * r0
RBUF = 2 * cutoff(sw_eq) *  ones(length(RDOM))

atmax, _ = SWqmmm.edge_config(Rmax, sw_eq)
AT, Iinmax, Iinnext = QMMM2.Solve.cluster_sequence(atmax, RDOM, RBUF)
UU, EE, Umax, Emax  =  QMMM2.Solve.solve_sequence(atmax, AT, Iinmax, Iinnext)
err2, errinf = QMMM2.Solve.errors(atmax, AT, Iinmax, UU, Umax)
Eerr = abs.(EE .- Emax)

##
plot(; xaxis = (:log, "domain radius [Å]"),
       yaxis = (:log, "errors"),
       title = "Domain Test, Si, SW",
       legend = :bottomleft )
plot!( RDOM, err2,   c=1, lw=3, m=:o, ms=8, label =  "energy-norm" )
plot!( RDOM, errinf, c=2, lw=3, m=:o, ms=8, label =  "max-norm (strains)" )
plot!( RDOM, Eerr,   c=3, lw=3, m=:o, ms=8, label =  "energy-diff. [eV]" )
t = RDOM[3:5]
plot!(t, 3*t.^(-1),  c=:black, lw=2, ls=:dot, label = "R^-1, R^-2")
plot!(t, 12*t.^(-2), c=:black, lw=2, ls=:dot, label = "")
