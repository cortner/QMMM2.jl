
using JuLIP, QMMM2, SHIPs, PrettyTables, LinearAlgebra, Plots
using JuLIP.MLIPs
include(@__DIR__() * "/swqmmm.jl")

##
sw = StillingerWeber()
r0 = rnn(:Si)
# domain radius of reference domain ("exact" solution)
Rmax = 16.0 * r0
# domain radius on approximation domains
RDOM = [2.0, 3.0, 5.0, 7.0, 9.0] * r0
RBUF = 2*cutoff(sw) * ones(length(RDOM))

atmax = SWqmmm.int_config(Rmax, sw)
AT, Iinmax, Iinnext = QMMM2.Solve.cluster_sequence(atmax, RDOM, RBUF)
UU, EE, Umax, Emax  =  QMMM2.Solve.solve_sequence(atmax, AT, Iinmax, Iinnext)
err2, errinf = QMMM2.Solve.errors(atmax, AT, Iinmax, UU, Umax)
Eerr = abs.(EE .- Emax)

##

plot(; xaxis = (:log, "domain radius [Å]"),
       yaxis = (:log, "errors"),
       title = "Domain Test, Si, SW, Interstitial",
       legend = :bottomleft )
plot!( RDOM, err2,   c=1, lw=3, m=:o, ms=8, label =  "energy-norm" )
plot!( RDOM, errinf, c=2, lw=3, m=:o, ms=8, label =  "max-norm (strains)" )
plot!( RDOM, Eerr,   c=3, lw=3, m=:o, ms=8, label =  "energy-diff. [eV]" )
t = RDOM[3:5]
t1 = RDOM[2:5]
plot!(t, 4.5*t.^(-1.5),  c=:black, lw=2, ls=:dot, label = "R^-1.5, R^-3")
plot!(t1, 6*t1.^(-3),  c=:black, lw=2, ls=:dot, label = "")
plot!(t, 30*t.^(-3), c=:black, lw=2, ls=:dot, label = "")
