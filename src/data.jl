using StaticArrays
using JuLIP, JuLIP.Potentials
using JuLIP.Potentials: site_energy, site_energy_d
using JuLIP: AbstractCalculator

function evec(i)
   if i == 1
      return SVector(1.0,0.0,0.0)
   elseif i == 2
      return SVector(0.0,1.0,0.0)
   elseif i == 3
      return SVector(0.0,0.0,1.0)
   end
   @error("`evec`: input must be 1,2 or 3")
end



function eval_dataset!(D::Dict, calc::AbstractCalculator; key = "train")
    at = Atoms(D["at"])
    for dat in D["data"]   # TODO: parallelise this loop!
       # get some info about the local lsq system
       dt = dat["datatype"]
       # evaluate this data point
       val = eval_dat(Val(Symbol(dt)), dat, calc, at)
       dat[key] = val
    end
    return D
end



"""
`eval_dat(::Val{SYM}, dat, calc, at) -> Vector{Float64}`

The following `SYM`s  are registered so far:
- `:E` : total energy
- `:F` : forces
- `:V` : virial
- `:Es` : site energy
- `:dEs` : site energy gradient
- `:d2Esh` : site energy hessian, FD approximation
- `:FC` : force-constants, i.e. a representation of the hessian
"""
function eval_dat end

eval_dat(::Val{:Es}, dat, calc, at) =
        site_energy(calc, at, 1)

eval_dat(::Val{:dEs}, dat, calc, at) =
        site_energy_d(calc, at, 1)

eval_dat(::Val{:d2Esh}, dat, calc, at) =
        _d2Esh(calc, at, 1, dat["l"], dat["i"], dat["h"])

eval_dat(::Val{:E}, dat, calc, at) =
        energy(calc, at)

eval_dat(::Val{:F}, dat, calc, at) =
        collect(mat(forces(calc, at))[:])

eval_dat(::Val{:V}, dat, calc, at) =
        virial(calc, at)[SVector(1,2,3,5,6,9)]

eval_dat(::Val{:FC}, dat, calc, at) =
        _force_constants(calc, at, dat["i0"], dat["h"], dat["i"])



# -----------------------------------------------------------------------------
#   SPECIAL EVALUATION ROUTINES FOR TB/QM MODELS
# -----------------------------------------------------------------------------


function eval_dataset_tb!(D::Dict, calc::AbstractCalculator; key = "train")
    at = Atoms(D["at"])
    DTs = [ dat["datatype"]  for dat in D["data"] ]
    for dt in unique(DTs)
        Idt = findall(DTs .== dt)
        eval_dataset_tb!(Val(Symbol(dt)), D.data[Idt], calc, at; key=key)
    end
    return D
end


function eval_dataset_tb!(valdt::Union{Val{:Es}, Val{:dEs}}, data, calc, at; key=key)
    @assert length(data) == 1
    data[1][key] = eval_dat(valdt, data[1], calc, at)
    return nothing
end


function eval_dataset_tb!(::Val{:d2Esh}, data, calc, at; key=key)
    h = data[1]["h"]
    # TODO: test all h are the same!
    for i = 1:3, sig in [1, -1]
        # perturb X

        # solve the eval problem

        # compute dEs on all neighbours we care about
        # as determined by data sketch
        for dat in data
            if dat["i"] == i
                l = dat["l"]
                @assert dat["h"] == h
                # compute dEs

                # write dEs into the data point
                calc[key] += dEs * sig
            end
        end
    end
    for dat in data
        calc[key] /= (2*h)
    end
end





"""
l0 : site that we are perturbing
 l : site at which the site energy is evaluated
 i : direction of the perturbation (E1, E2, E3)
 h : finite-difference step
"""
function _d2Esh(calc, at::Atoms,
                l0::Integer, l::Integer, i::Integer, h::Real)
   X = positions(at)
   X[l0] += h * evec(i)
   set_positions!(at, X)
   dVp = site_energy_d(calc, at, l)
   X[l0] -= 2*h * evec(i)
   set_positions!(at, X)
   dVm = site_energy_d(calc, at, l)
   X[l0] += h * evec(i)
   set_positions!(at, X)
   return (dVp - dVm) / (2*h)
end

function _force_constants(calc, at::Atoms, i0, h, i)
    at[i0] += h * evec(i)
    Fp = forces(calc, at)
    at[i0] -= 2 * h * evec(i)
    Fm = forces(calc, at)
    at[i0] +=  h * evec(i0)
    return  mat( (Fp - Fm) / (2*h) )[:] |> collect
end
