
module Solve

using JuLIP, QMMM2, LinearAlgebra
using JuLIPMaterials, StaticArrays
using JuLIP.MLIPs

function autocell12(atref, atnew)
   C = Matrix(cell(atref))
   x, y, _ =  xyz(atnew)
   @assert minimum(x) >= 0
   @assert minimum(y) >= 0
   C[1,1] = maximum(x)
   C[2,2] = maximum(y)
   return C
end


"""
`function cluster_sequence(atmax, RDOM, RBUF,

* `get_data(atmax, "xcore")` must return the position of the defect core

"""
function cluster_sequence(atmax, RDOM, RBUF,
                          dims = findall(.!(pbc(atmax))))
   @assert sort(RDOM) == RDOM
   X_max = positions(atmax)
   x0 = get_data(atmax, "xcore")::JVecF

   Iinmax = Vector{Vector{Int}}(undef, length(RDOM))
   Iinnext = Vector{Vector{Int}}(undef, length(RDOM))
   AT = Vector{Atoms{Float64}}(undef, length(RDOM))

   at_next = atmax
   Iinmax_next = 1:length(atmax)

   for n = length(RDOM):-1:1
      X_next = positions(at_next)
      r_next = [ norm(x[dims] - x0[dims]) for x in X_next ]
      Iinnext[n] =  findall( r_next .<= RDOM[n] + RBUF[n] )
      Iinmax[n] = Iinmax_next[Iinnext[n]]
      Iinmax_next = Iinmax[n]
      at_next = Atoms(:Si, X_next[Iinnext[n]])
      empty!(at_next.Z)
      append!(at_next.Z, fill(atomic_number(:Si), length(at_next)) )
      r = r_next[Iinnext[n]]
      set_free!(at_next, findall(r .<=  RDOM[n]))
      set_pbc!(at_next, pbc(atmax))
      set_cell!(at_next, autocell12(atmax, at_next))
      AT[n] = at_next
   end

   # test this construction
   for n = 1:length(RDOM)-1
      at = AT[n]
      X = positions(at)
      at_next = AT[n+1]
      X_next = positions(at_next)
      @assert X == X_max[Iinmax[n]]
      @assert X == X_next[Iinnext[n]]
   end
   X = positions(AT[end])
   @assert X ==  X_max[Iinmax[end]]

   return AT, Iinmax, Iinnext
end

function solve_sequence(atmax, AT, Iinmax, Iinnext,
                        calc = calculator(atmax),
                        preconcalc = calc)
   UU = Vector{Vector{JVecF}}(undef, length(AT))
   EE = Vector{Float64}(undef, length(AT))

   # solve the first one
   @show length(AT[1])
   set_calculator!(AT[1], calc)
   X0 = positions(AT[1])
   E0 = energy(AT[1])
   minimise!(AT[1]; verbose=2) #, precond=FF(AT[1]))
   UU[1] = positions(AT[1]) .- X0
   EE[1] = energy(AT[1]) - E0

   # now solve sequentially
   for n = 2:length(AT)
      @show length(AT[n])
      set_calculator!(AT[n], calc)
      E0 = energy(AT[n])
      X0 = positions(AT[n])
      X = copy(X0)
      X[Iinnext[n-1]] += UU[n-1]
      set_positions!(AT[n], X)
      minimise!(AT[n]; verbose=2) #, precond=FF(AT[n], preconcalc))
      UU[n] = positions(AT[n]) - X0
      EE[n] = energy(AT[n]) - E0
   end

   # and the last one - the reference calculation
   @show length(atmax)
   set_calculator!(atmax, calc)
   X0max = positions(atmax)
   E0max = energy(atmax)
   X = copy(X0max)
   X[Iinmax[end]] += UU[end]
   set_positions!(atmax, X)
   minimise!(atmax; verbose=2) # , precond=FF(atmax, preconcalc))
   Umax = positions(atmax) - X0max
   Emax = energy(atmax) - E0max

   return UU, EE, Umax, Emax
end


function errors(atmax, AT, Iinmax, UU, Umax; rcut=3.5)
   errs2 = zeros(length(AT))
   errsinf = zeros(length(AT))
   for n = 1:length(AT)
      at = AT[n]
      Uerr = copy(Umax)
      Uerr[Iinmax[n]] -= UU[n]
      ee = JuLIPMaterials.strains(Uerr, atmax, rcut = rcut)
      errs2[n] = norm(ee, 2)
      errsinf[n] = norm(ee, Inf)
   end
   return errs2, errsinf
end

end
