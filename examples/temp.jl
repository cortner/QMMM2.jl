
using JuLIP, QMMM2, SHIPs
using JuLIP.MLIPs
using NeighbourLists, LinearAlgebra

# fname = @__DIR__() * "/C_test_data.json"
fname = "/Users/ortner/Dropbox/Work/Projects/SiteEFitting/data/C_Dim3.json"
D = load_json(fname)
sym = :C  # => can get this from atoms object

trans = PolyTransform(2, rnn(sym))
basis = IPSuperBasis(OneBody(1.0),
                     SHIPBasis(3, 15, 1.5, trans, 2, 0.7*rnn(:C), 5.0) )
length(basis)
# weights = Dict("Es" => 100.0, "dEs" => 10.0, "d2Es" => 1.0, "d3Es" => 0.1)
weights = Dict("Es" => 0.1, "dEs" => 0.1, "d2Esh" => 10.0, "d3Esh" => 0.0)

##

🚢, fitinfo = QMMM2.lsqfit(basis, D, weights; rtol=1e-6)

QMMM2.print_errors(fitinfo)

dat = D["data"][3]
d2Esh = hcat(dat["d2Esh"]...)
at = Atoms(D["at"])
fit = QMMM2._site_energy_d2h(🚢, at, 1, dat["l"], dat["i"], dat["h"])
Ilge = findall( sqrt.(sum(abs2, d2Esh - fit; dims=1))[:] .> 0.15 )

display([d2Esh; fit][:, Ilge])

🚢

positions(at)[Ilge] |> mat
dat


norm(d2Esh - fit, Inf)
