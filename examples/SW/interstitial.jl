
using JuLIP, QMMM2, SHIPs, PrettyTables, LinearAlgebra, Plots
using JuLIP.MLIPs
include(@__DIR__() * "/swqmmm.jl")

function int_config(Rdom, Rbuf = 2 * cutoff(StillingerWeber()))
   # -----------------------------------
   #  construct the configuration
   at = cluster(:Si, Rdom+Rbuf)
   X = positions(at)
   i0 = 1
   x0 = X[i0]
   r = [ norm(x - x0) for x in X]
   Ineig = findall(r .< 3.0)
   Rneig = [ x - x0 for x in X[Ineig] ]
   rup = [1.3575, 1.3575, 1.3575]
   if rup ∈ Rneig
      σ = 1.0
   else
      σ = -1.0
   end
   rleft = σ * [-1.3575, -1.3575, 1.3575]
   dint = 0.5 * (norm(rup) - dot(rleft, rup/norm(rup)))
   xint = x0 - σ * rup * dint
   at = append(at, [xint])
   # hack at.Z => TODO in JuLIP
   empty!(at.Z)
   append!(at.Z, fill(atomic_number(:Si), length(at)) )
   # -----------------------------------
   #  compute the domains
   X = positions(at)
   r = [ norm(x - x0) for x in X]
   set_free!(at, findall(r .<= Rdom))
   set_data!(at, "xcore", x0)
   return at
end
#
# function cluster_sequence(Rmax, RDOM,
#                           Rbufmax =  2 * cutoff(StillingerWeber()),
#                           RBUF = fill(2 * cutoff(StillingerWeber()), length(RDOM)))
#    @assert sort(RDOM) == RDOM
#    atmax = int_config(Rmax, Rbufmax)
#    X_max = positions(atmax)
#    x0 = positions(atmax)[1]
#
#    Iinmax = Vector{Vector{Int}}(undef, length(RDOM))
#    Iinnext = Vector{Vector{Int}}(undef, length(RDOM))
#    AT = Vector{Atoms{Float64}}(undef, length(RDOM))
#
#    at_next = atmax
#    Iinmax_next = 1:length(atmax)
#
#    for n = length(RDOM):-1:1
#       X_next = positions(at_next)
#       r_next = [ norm(x - x0) for x in X_next ]
#       Iinnext[n] =  findall( r_next .<= RDOM[n] + RBUF[n] )
#       Iinmax[n] = Iinmax_next[Iinnext[n]]
#       Iinmax_next = Iinmax[n]
#       at_next = Atoms(:Si, X_next[Iinnext[n]])
#       empty!(at_next.Z)
#       append!(at_next.Z, fill(atomic_number(:Si), length(at_next)) )
#       r = r_next[Iinnext[n]]
#       set_free!(at_next, findall(r .<=  RDOM[n]))
#       AT[n] = at_next
#    end
#
#    # test this construction
#    for n = 1:length(RDOM)-1
#       at = AT[n]
#       X = positions(at)
#       at_next = AT[n+1]
#       X_next = positions(at_next)
#       @assert X == X_max[Iinmax[n]]
#       @assert X == X_next[Iinnext[n]]
#    end
#    X = positions(AT[end])
#    @assert X ==  X_max[Iinmax[end]]
#
#    return atmax, AT, Iinmax, Iinnext
# end
#
# function solve_sequence(atmax, AT, Iinmax, Iinnext)
#    sw = StillingerWeber()
#    UU = Vector{Vector{JVecF}}(undef, length(AT))
#    EE = Vector{Float64}(undef, length(AT))
#
#    # solve the first one
#    @show length(AT[1])
#    set_calculator!(AT[1], sw)
#    X0 = positions(AT[1])
#    E0 = energy(AT[1])
#    minimise!(AT[1]; verbose=2, precond=FF(AT[1]))
#    UU[1] = positions(AT[1]) .- X0
#    EE[1] = energy(AT[1]) - E0
#
#    # now solve sequentially
#    for n = 2:length(AT)
#       @show length(AT[n])
#       set_calculator!(AT[n], sw)
#       E0 = energy(AT[n])
#       X0 = positions(AT[n])
#       X = copy(X0)
#       X[Iinnext[n-1]] += UU[n-1]
#       set_positions!(AT[n], X)
#       minimise!(AT[n]; verbose=2, precond=FF(AT[n]))
#       UU[n] = positions(AT[n]) - X0
#       EE[n] = energy(AT[n]) - E0
#    end
#
#    # and the last one - the reference calculation
#    @show length(atmax)
#    set_calculator!(atmax, sw)
#    X0max = positions(atmax)
#    E0max = energy(atmax)
#    X = copy(X0max)
#    X[Iinmax[end]] += UU[end]
#    set_positions!(atmax, X)
#    minimise!(atmax; verbose=2, precond=FF(atmax))
#    Umax = positions(atmax) - X0max
#    Emax = energy(atmax) - E0max
#
#    return UU, EE, Umax, Emax
# end

##
r0 = rnn(:Si)
# domain radius of reference domain ("exact" solution)
Rmax = 16.0 * r0
# domain radius on approximation domains
RDOM = [2.0, 3.0, 5.0, 7.0, 9.0] * r0
RBUF = 2*cutoff(StillingerWeber()) * ones(length(RDOM))

atmax = int_config(Rmax)
set_calculator!(atmax, StillingerWeber())
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


##
#
# # get the QM and MM potentials
# Vmm, _ = SWqmmm.train_ship(3, 5, rtol=2e-3)
# Vqm = sw = StillingerWeber()
#
# # get a computational domain
# r0 = rnn(:Si)
# at, Iqm = int_config(10 * r0, 5 * r0)
# at = prepare_qmmm!(at, EnergyMixing;
#                    Vqm = Vqm, Vmm = Vmm, Iqm = Iqm)
# Vh = calculator(at)
#
#
# # solve the optimisation problem
# set_calculator!(at, sw)
# minimise!(at; precond=FF(at, sw), verbose=2
#       )
#
#
# set_calculator!(at, Vh)
# minimise!(at; precond=FF(at, sw), verbose=2
#       )
#

# 1. convergence of plain SW test
# add QM region to the test
