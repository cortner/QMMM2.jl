# create the data from forces and train the MM potential for QMMM simulations

# using QMMM2
using JuLIP, SHIPs, PrettyTables, LinearAlgebra, Plots,
      DataFrames, JuLIPMaterials
using JuLIP.MLIPs
using SKTB
using SKTB: ZeroTGrand, ZeroT, MPGrid, spectrum, GammaPoint

include(@__DIR__() * "/NRLqmmm.jl")
# include(@__DIR__() * "/examples/NRL/NRLqmmm.jl")

TB = SKTB
NRLTB = SKTB.NRLTB


## QM model: NRL-TB
eF = 5.5212408; # for Si
tbm = NRLTB.NRLTBModel(:Si,
                       ZeroTGrand(eF = eF),
                       bzquad = TB.GammaPoint(),
                       cutoff = :original );
r0 = rnn(:Si)
# QM cutoff: from the locality
tbm_cutoff_d1 = 5.0 * r0
tbm_cutoff_d2 = 3.0 * r0
tbm_cutoff_d3 = 3.0 * r0

## create the database from the QM (NRL-TB) model
train_size = 5;
at_train = bulk(:Si, cubic=true) * (train_size, train_size, 1)
rcut = [tbm_cutoff_d1, tbm_cutoff_d2, tbm_cutoff_d3]
at = at_train
D = QMMM2.data_djF_sketch(at, 1e-3, rcut)
QMMM2.eval_dataset_tb!(D, tbm; key="NRLTB")
# QMMM2.eval_dataset!(D, tbm; key="NRLTB")

## train the MM model 🚢
wL = 1.7          # 1.0 1.5  1.75
rin = 0.7         # 0.6 0.7 0.8
rtol = 1e-3       # 1e-5, 1e-4, ...
bo = 2            # 2, 3, 4, 5
deg = 18           # 5, 10, 12, 15
weights = Dict("F" => 10.0, "FC" => 1.0)
basis = NRLqmmm.get_basis(bo, deg; rinfact=rin, wL=wL)
@show length(basis)
# TODO: test the parameters
🚢, fitinfo = NRLqmmm.train_ship(D, bo, deg,
               wL = wL, rinfact = rin, weights = weights, rtol = rtol)
@show fitinfo["rmse"]
