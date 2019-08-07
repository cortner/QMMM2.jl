

module SWqmmm

using JuLIP, QMMM2, NeighbourLists, LinearAlgebra, SHIPs, JuLIPMaterials
using JuLIP.MLIPs: IPSuperBasis

pot() = StillingerWeber()
at_train() = bulk(:Si, cubic=true) * 2

function sketch()
   sw =  pot()
   rcut = cutoff(sw)
   at = at_train()
   return QMMM2.data_djEs_sketch(at, 1e-3, [rcut, rcut, rcut])
end

function training_set()
   sw = pot()
   sw_sketch = sketch()
   QMMM2.eval_dataset!(sw_sketch, sw; key="sw")
end

function get_basis(bo, deg; rinfact = 0.8, wL = 1.5)
   sw = pot()
   r0 = rnn(:Si)
   rcut = cutoff(sw)
   rin = rinfact * r0
   trans = PolyTransform(2, rnn(:Si))
   fcut = PolyCutoff1s(2, rin, rcut)
   shipB = SHIPBasis(SparseSHIP(bo, :Si, deg, wL), trans, fcut)
   basis = IPSuperBasis(OneBody(1.0), shipB)
end

function train_ship(bo, deg;
            wL = 1.5,
            rinfact = 0.8,
            weights = Dict("Es" => 100.0, "dEs" => 10.0, "d2Esh" => 1.0),
            rtol = 1e-5
         )
   trainset = training_set()
   basis = get_basis(bo, deg; rinfact=rinfact, wL=wL)
   🚢, fitinfo = QMMM2.lsqfit(basis, trainset, weights; rtol=rtol, key="sw")
   @show fitinfo["rmse"]
   return 🚢, fitinfo
end


function int_config(Rdom, calc = StillingerWeber(), Rbuf = 2 * cutoff(calc))
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
   set_calculator!(at, calc)
   return at
end



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
   set_calculator!(at, calc)
   return at, Ifree
end

end


# # test convergence
# for deg = 2:6
#    🚢, fitinfo = SWqmmm.train_ship(3, deg)
# end
