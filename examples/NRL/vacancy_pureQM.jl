# convergence tests for pure NRL-TB

using JuLIP, QMMM2, SHIPs, PrettyTables, LinearAlgebra, Plots
using JuLIP.MLIPs
using SKTB
using SKTB: ZeroTGrand, ZeroT, MPGrid, spectrum, GammaPoint

include(@__DIR__() * "/NRLqmmm.jl")
TB = SKTB
NRLTB = SKTB.NRLTB

## calc
eF = 5.5212408; # for Si
# eF = 10.89102; # for C
tbm = NRLTB.NRLTBModel(:Si,
                       ZeroTGrand(eF = eF),
                       bzquad = TB.GammaPoint(),
                       cutoff = :original );
r0 = rnn(:Si)
tbm_cutoff = 2.0 * r0   # from the locality of to accuracy 1.0e-3

# domain radius of reference domain ("exact" solution)
Rmax = 5.0 * r0
# domain radius on approximation domains
RDOM = [2.0, 3.0, 4.0] * r0
# RBUF = 2.0 * cutoff(tbm.H) * ones(length(RDOM))
RBUF = tbm_cutoff * ones(length(RDOM))

# atomic configuration
atmax = NRLqmmm.int_config(Rmax, tbm, tbm_cutoff)

# solve the equilibrium
AT, Iinmax, Iinnext = QMMM2.Solve.cluster_sequence(atmax, RDOM, RBUF)
UU, EE, Umax, Emax  = QMMM2.Solve.solve_sequence(atmax, AT, Iinmax, Iinnext)
err2, errinf = QMMM2.Solve.errors(atmax, AT, Iinmax, UU, Umax)
Eerr = abs.(EE .- Emax)

## output
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
