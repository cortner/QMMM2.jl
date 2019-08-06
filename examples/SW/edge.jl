
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
