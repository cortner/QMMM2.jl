
using JuLIP, QMMM2, SHIPs, PrettyTables
using JuLIP.MLIPs
using NeighbourLists, LinearAlgebra

# fname = @__DIR__() * "/C_test_data.json"
fname = "/Users/ortner/Dropbox/Work/Projects/SiteEFitting/data/C_Dim3.json"
D = load_json(fname)

trans = PolyTransform(2, rnn(:C))
weights = Dict("Es" => 0.01, "dEs" => 10.0, "d2Esh" => 1.0)

err0 = Float64[]
err1 = Float64[]
err2 = Float64[]

@info("4B BASIS TEST")
for deg in [6, 8, 10, 12, 14, 16 ] #, 18, 20]
   basis = IPSuperBasis(OneBody(1.0),
                        SHIPBasis(3, deg, 1.0, trans, 2, 0.7*rnn(:C), 5.0) )
   @info("deg = $deg, len(B) = $(length(basis))")
   🚢, fitinfo = QMMM2.lsqfit(basis, D, weights; rtol=1e-5)
   @show fitinfo["maxe"]
   push!(err0, fitinfo["maxe"]["Es"])
   push!(err1, fitinfo["maxe"]["dEs"])
   push!(err2, fitinfo["maxe"]["d2Esh"])
end

@info("Summary of Errors:")
pretty_table([err0 err1 err2], ["Es", "dEs", "d2Es"],
             formatter=ft_printf("%.3e"))


# @info("5B BASIS TEST")
# for deg in [6, 8, 10, 12, 13, 14 ]
#    basis = IPSuperBasis(OneBody(1.0),
#                         SHIPBasis(4, deg, 1.5, trans, 2, 0.7*rnn(:C), 5.0) )
#    @info("deg = $deg, len(B) = $(length(basis))")
#    🚢, fitinfo = QMMM2.lsqfit(basis, D, weights; rtol=1e-5)
#    @show fitinfo["maxe"]
#    push!(err0, fitinfo["maxe"]["Es"])
#    push!(err1, fitinfo["maxe"]["dEs"])
#    push!(err2, fitinfo["maxe"]["d2Esh"])
# end
#
# @info("Summary of Errors:")
# pretty_table([err0 err1 err2], ["Es", "dEs", "d2Es"],
#              formatter=ft_printf("%.3e"))
