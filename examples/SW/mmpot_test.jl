
using JuLIP, QMMM2, SHIPs, PrettyTables, LinearAlgebra, Plots,
      DataFrames
using JuLIP.MLIPs
include(@__DIR__() * "/swqmmm.jl")


## reference model and MM potential(s)
r0 = rnn(:Si)
sw = StillingerWeber()


## test convergence
ships =  []
degs = [2, 3, 4, 5, 6, 7, 8]
rmse_E = []
rmse_DE = []
rmse_D2E = []
for deg = degs
   🚢, fitinfo = SWqmmm.train_ship(3, deg, wL=1.0, rinfact=0.6, rtol=5e-4)
   push!(ships, 🚢)
   push!(rmse_E, fitinfo["rmse"]["Es"])
   push!(rmse_DE, fitinfo["rmse"]["dEs"])
   push!(rmse_D2E, fitinfo["rmse"]["d2Esh"])
end
println(); println()

println("RMSE-Table")
df_rmse = DataFrame(deg = degs, E = rmse_E, dE = rmse_DE, d2E = rmse_D2E)
display(df_rmse)
