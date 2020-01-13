module NRLqmmm

# using QMMM2
QMMM2 = Main.QMMM2

using JuLIP, NeighbourLists, LinearAlgebra, SHIPs, JuLIPMaterials
using JuLIP.MLIPs: IPSuperBasis
using SKTB
using SKTB: ZeroTGrand, ZeroT, MPGrid, spectrum, GammaPoint

TB = SKTB
NRLTB = SKTB.NRLTB


# TODO: rcut = 5.5 according to G.Csany
function get_basis(bo, deg; rinfact = 0.8, wL = 1.5, rcut = 6.0)
   r0 = rnn(:Si)
   rin = rinfact * r0
   trans = PolyTransform(2, rnn(:Si))
   fcut = PolyCutoff1s(2, rin, rcut)
   spec = SparseSHIP(:Si, bo, deg; wL = wL)
   shipB = SHIPBasis(spec, trans, fcut)
   basis = IPSuperBasis(OneBody(1.0), shipB)
end


function train_ship(train_database, bo, deg;
            wL = 1.5,
            rinfact = 0.8,
            # weights = Dict("Es" => 1.0, "dEs" => 10.0, "d2Esh" => 1.0),
weights = Dict("Es" => 1.0, "dEs" => 1.0, "d2Esh" => 1.0, "d3Esh" => 1.0),
            rtol = 1e-5,
            verbose = true
         )
   trainset = train_database
   basis = get_basis(bo, deg; rinfact=rinfact, wL=wL)
   🚢, fitinfo = QMMM2.lsqfit(basis, trainset, weights;
                              pqrtol=rtol, key="NRLTB", verbose=verbose)
   @show fitinfo["rmse"]
   return 🚢, fitinfo
end


# construct the configuration with a vacancy
# 2D with p.b.c. in the 3rd direction
function vac2d_config(Ndom, calc)
   at = bulk(:Si, cubic=true) * (Ndom, Ndom, 1)
   X0 = positions(at)
   coord_centre = [at.cell[1]/2.0, at.cell[5]/2.0, at.cell[9]/2.0]
   r0 = [ norm(x - coord_centre) for x in X0]
   i0 = findmin(r0)[2]
   x0 = X0[i0]
   # deleteat!(at, i0)
   # compute the domains
   X = positions(at)
   set_data!(at, "xcore", x0)
   set_calculator!(at, calc)
   # inplane!(at)
   mask = fill(false, 3, length(at))
   free = 1:length(at)
   mask[1, free] .= true
   mask[2, free] .= true
   set_mask!(at, mask)
   return at
end


# construct the configuration with a vacancy, 3D cluster
function vac_config(Rdom, calc, Rbuf)
   at = cluster(:Si, Rdom+Rbuf)
   at0 = deepcopy(at)
   X0 = positions(at0)
   i0 = 1
   x0 = X0[i0]
   deleteat!(at, i0)
   empty!(at.Z)
   append!(at.Z, fill(atomic_number(:Si), length(at)) )
   #  compute the domains
   X = positions(at)
   r = [ norm(x - x0) for x in X]
   set_free!(at, findall(r .<= Rdom))
   set_data!(at, "xcore", x0)
   set_calculator!(at, calc)
   return at
end


#  construct the configuration with a edge dislocation
function edge_config(Rdom, calc = sw_eq, Rbuf = 2 * cutoff(calc))
   at, x0 = JuLIPMaterials.Si.edge110(:Si, Rdom + Rbuf;
               calc=calc, truncate=true, sym=true)
   set_data!(at, "xcore", x0)
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
