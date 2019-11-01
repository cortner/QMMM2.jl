module NRLqmmm

using JuLIP, QMMM2, NeighbourLists, LinearAlgebra, SHIPs, JuLIPMaterials
using JuLIP.MLIPs: IPSuperBasis
using SKTB
using SKTB: ZeroTGrand, ZeroT, MPGrid, spectrum, GammaPoint

TB = SKTB
NRLTB = SKTB.NRLTB

## QM model
eF = 5.5212408; # for Si
tbm = NRLTB.NRLTBModel(:Si,
                       ZeroTGrand(eF = eF),
                       bzquad = TB.GammaPoint(),
                       cutoff = :original );
# QM cutoff: from the locality of to accuracy 1.0e-3
r0 = rnn(:Si)
tbm_cutoff_1 = 3.0 * r0
tbm_cutoff_2 = 2.0 * r0
tbm_cutoff_3 = 1.0 * r0
# trained system
at_train() = bulk(:Si, cubic=true) * 4

function sketch()
   rcut = [tbm_cutoff_1, tbm_cutoff_2, tbm_cutoff_3]
   at = at_train()
   return QMMM2.data_djEs_sketch(at, 1e-3, rcut)
end

function training_set()
   tbm_sketch = sketch()
   # QMMM2.eval_dataset!(tbm_sketch, tbm; key="NRLTB")
   QMMM2.eval_dataset_tb!(tbm_sketch, tbm; key="NRLTB")
end

function get_basis(bo, deg; rinfact = 0.8, wL = 1.5)
   r0 = rnn(:Si)
   rcut = tbm_cutoff
   rin = rinfact * r0
   trans = PolyTransform(2, rnn(:Si))
   fcut = PolyCutoff1s(2, rin, rcut)
   spec = SparseSHIP(:Si, bo, deg; wL = wL)
   shipB = SHIPBasis(spec, trans, fcut)
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
   🚢, fitinfo = QMMM2.lsqfit(basis, trainset, weights; rtol=rtol, key="NRLTB")
   @show fitinfo["rmse"]
   return 🚢, fitinfo
end


function int_config(Rdom, calc, Rbuf)
   # -----------------------------------
   #  construct the configuration with a vacancy
   at = cluster(:Si, Rdom+Rbuf)

   at0 = deepcopy(at)
   X0 = positions(at0)
   i0 = 1
   x0 = X0[i0]
   # r = [ norm(x - x0) for x in X0]
   # Ineig = findall(r .< 3.0)
   # Rneig = [ x - x0 for x in X0[Ineig] ]
   # rup = [1.3575, 1.3575, 1.3575]
   # if rup ∈ Rneig
   #    σ = 1.0
   # else
   #    σ = -1.0
   # end
   # rleft = σ * [-1.3575, -1.3575, 1.3575]
   # dint = 0.5 * (norm(rup) - dot(rleft, rup/norm(rup)))
   # displ = [1.2, 0.0, 0.0]
   # println(x0)
   # xint1 = x0 - displ
   # xint2 = x0 + displ

   deleteat!(at, i0)
   # at = append(at, [xint1, xint2])

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
