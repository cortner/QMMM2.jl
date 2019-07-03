
using JuLIP, QMMM2, SHIPs
using NeighbourLists

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
basis = SHIPBasis(3, 15, 1.5, trans, 2, 0.7*rnn(:C), 4.0)

weights = Dict("Es" => 100.0, "dEs" => 10.0, "d2Es" => 1.0)

🚢 = QMMM2.lsqfit(basis, D, weights)


dat1["Es"]
energy(🚢, at) / length(at)
site_energy(🚢, at, 1)
site_energy_d(🚢, at, 1)

Rs = rand(JVecF, 46)  * 3
tmp = SHIPs.alloc_temp_d(🚢, Rs)
dEs = zeros(JVecF, 46)
using BenchmarkTools
@btime SHIPs.evaluate_d!($dEs, $🚢, $Rs, $tmp)

atu = bulk(:C)
energy(🚢, atu) / 2
energy(🚢, atu) / length(atu)
virial(🚢, atu)
