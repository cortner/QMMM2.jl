
using JuLIP, QMMM2, SHIPs
using JuLIP.MLIPs
using NeighbourLists, LinearAlgebra

fname = @__DIR__() * "/C_test_data.json"
D = load_json(fname)

# fix the database
at = Atoms(D["at"])
h = D["h"]
data = D["data"]
dat1 = Dict( "Es" => data[1]["Es"] )
dat2 = Dict( "dEs" => data[1]["dEs"] )
D["data"] = [ [dat1, dat2]; data[2:end] ]

trans = PolyTransform(2, rnn(:C))
basis = IPSuperBasis(OneBody(1.0),
                     SHIPBasis(3, 18, 1.5, trans, 2, 0.7*rnn(:C), 5.0) )
length(basis)
weights = Dict("Es" => 100.0, "dEs" => 100.0, "d2Es" => 1.0)

🚢 = QMMM2.lsqfit(basis, D, weights; rtol=1e-5)

dat1["Es"] - site_energy(🚢, at, 1)
errdEs = dat2["dEs"] - site_energy_d(🚢, at, 1)
maximum(norm.(errdEs, Inf))

atu = bulk(:C) * 3
energy(🚢, atu) / length(atu)
site_energy(🚢, atu, 1)
