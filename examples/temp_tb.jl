

using JuLIP, SKTB

at = bulk(:Si)
tb = SKTB.Kwon.KwonTBModel(; potential = SKTB.ZeroT(),
                             bzquad = SKTB.MPGrid(at, (10,10,10)) )
energy(tb, at)
F0 = defm(at)
ss = range(1.0, 1.4, length = 100)
EE = Float64[]
for s in ss
   F = s * F0
   set_defm!(at, F, updatepositions=true)
   push!(EE, energy(tb, at))
end

using Plots
plot(ss, EE)

findmin(EE)
ss[67]
