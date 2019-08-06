

module SWqmmm

using JuLIP, QMMM2, NeighbourLists, LinearAlgebra, SHIPs
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

function get_basis(bo, deg; rinfact = 0.85, wL = 1.5)
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
            rinfact = 0.85,
            weights = Dict("Es" => 100.0, "dEs" => 10.0, "d2Esh" => 1.0),
            rtol = 1e-5
         )
   trainset = training_set()
   basis = get_basis(bo, deg; rinfact=rinfact, wL=wL)
   🚢, fitinfo = QMMM2.lsqfit(basis, trainset, weights; rtol=rtol, key="sw")
end

end


# # test convergence
# for deg = 2:6
#    🚢, fitinfo = SWqmmm.train_ship(3, deg)
# end
