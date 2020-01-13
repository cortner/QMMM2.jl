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


function eval_dataset!(D::Dict, calc::AbstractCalculator; key="train")
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

eval_dat(::Val{:E}, dat, calc, at) = energy(calc, at)

eval_dat(::Val{:F}, dat, calc, at) = forces(calc, at)
        # collect(mat(forces(calc, at))[:])

eval_dat(::Val{:FC}, dat, calc, at) =
        _force_constants(calc, at, 1, dat["i"], dat["h"])

eval_dat(::Val{:V}, dat, calc, at) =
        virial(calc, at)[SVector(1,2,3,5,6,9)]

# eval_dat(::Val{:EF}, dat, calc, at) = forces(calc, at)

# eval_dat(::Val{:EFV},dat,calc,at) = virial(calc,at)[SVector(1,2,3,5,6,9)]


# -----------------------------------------------------------------------------
#   SPECIAL EVALUATION ROUTINES FOR TB/QM MODELS
# -----------------------------------------------------------------------------

function eval_dataset_tb!(D::Dict, calc::AbstractCalculator; key="train")
    at = Atoms(D["at"])
    DTs = [ dat["datatype"]  for dat in D["data"] ]
    for dt in unique(DTs)
        Idt = findall(DTs .== dt)
        eval_dataset_tb!(Val(Symbol(dt)), D["data"][Idt], calc, at; key=key)
    end
    return D
end

function eval_dataset_tb!(valdt::Union{Val{:E},Val{:Es},Val{:F},Val{:dEs}},
                          data, calc, at; key="train")
    @assert length(data) == 1
    println("on reference: calculate ", valdt)
    data[1][key] = eval_dat(valdt, data[1], calc, at)
end

function eval_dataset_tb!(::Val{:d2Esh}, data, calc, at; key="train")
    l0 = 1
    X = positions(at) |> mat;
    h = data[1]["h"]
    # d2Esh = 1/2h * ( E_{ℓ,n}(y+h⋅e0) - E_{ℓ,n}(y-h⋅e0) )
    for i = 1:3, sig in [1, -1]
        println("perturb the ", l0, "-th atom in direction ", sig, i)
        X[i,l0] += sig * h
        atd = deepcopy(at)
        set_positions!(atd, X)
        X[i,l0] -= sig * h
        # compute dEs on all neighbours as determined by data sketch
        for dat in data
            if dat["i"] == i
                ℓ = dat["l"]
                @assert dat["h"] == h  # test all h are the same!
                println("calculating dE_", ℓ)
                dEs = site_energy_d(calc, atd, ℓ)
                # write dEs into the data point
                if !haskey(dat, key)
                    dat[key] = zeros(size(dEs))
                end
                dat[key] += dEs * sig / (2.0*h)
            end
        end
    end
end

function eval_dataset_tb!(::Val{:d3Esh}, data, calc, at; key="train")
    l0 = 1
    X = positions(at) |> mat;
    h = data[1]["h"]
    # collect all the "k" indecies
    Dks = [ dat["k"]  for dat in data ]
    for k in unique(Dks)
        Idk = findall(Dks .== k)
        # d3Esh = 1/4h²⋅[ ∇E_n(y+h⋅eⁱ_ℓ+h⋅eʲ_k) + ∇E_n(y-h⋅eⁱ_ℓ-h⋅eʲ_k)
        #                -∇E_n(y+h⋅eⁱ_ℓ) - ∇E_n(y+h⋅eʲ_k) ]
    for i_l0 = 1:3, sig_l0 in [1,-1], i_k = 1:3, sig_k in [-1,1]
        println("perturb the ", l0, "-th atom in direction ", i_l0,
                "with", sig_l0, "and perturb the ", k,
                "-th atom in direction ", i_k, "with", sig_k)
        X[i_l0,l0] += sig_l0 * h
        X[i_k,k] += sig_k * h
        atd = deepcopy(at)
        set_positions!(atd, X)
        X[i_l0,l0] -= sig_l0 * h
        X[i_k,k] -= sig_k * h
        # compute dEs on all neighbours as determined by data sketch
        for dat in data[Idk]
        if dat["i_l0"] == i_l0 && dat["i_k"] == i_k
            ℓ = dat["l"]
            @assert dat["h"] == h  # test all h are the same!
            println("calculating dE_", ℓ)
            dEs = site_energy_d(calc, atd, ℓ)
            # write dEs into the data point
            if !haskey(dat, key)
                dat[key] = zeros(size(dEs))
            end
            dat[key] += dEs * sig_l0*sig_k / (4.0*h^2)
        end
        end
    end
    end
end


function eval_dataset_tb!(::Val{:FC}, data, calc, at; key="train")
    l0 = 1
    X = positions(at) |> mat;
    h = data[1]["h"]
    # dFh = 1/2h * ( F_{,n}(y+h*e₀) - F_{,n}(y-h*e₀) )
    for i = 1:3, sig in [1, -1]
        println("perturb the origin atom in direction ", i, " ", sig*h)
        X[i,l0] += sig * h
        atd = deepcopy(at)
        set_positions!(atd, X)
        X[i,l0] -= sig * h
        # compute dEs on all neighbours as determined by data sketch
        for dat in data
            if dat["i"] == i
                @assert dat["h"] == h  # test all h are the same!
                println("calculating the forces\n")
                frc = forces(calc, atd)
                # write dEs into the data point
                if !haskey(dat, key)
                    dat[key] = zeros(size(frc))
                end
                dat[key] += frc * sig / (2.0*h)
            end
        end
    end
end

function eval_dataset_tb!(::Val{:d2Fh}, data, calc, at; key="train")
    l0 = 1
    X = positions(at) |> mat;
    h = data[1]["h"]
    # collect all the "k" indecies
    Dks = [ dat["k"]  for dat in data ]
    for k in unique(Dks)
        Idk = findall(Dks .== k)
        # d2Fh = 1/4h²⋅[ F(y+h⋅eⁱ_ℓ+h⋅eʲ_k) + F(y-h⋅eⁱ_ℓ-h⋅eʲ_k)
        #              - F(y+h⋅eⁱ_ℓ) - F(y+h⋅eʲ_k) ]
    for i_l0 = 1:3, sig_l0 in [1,-1], i_k = 1:3, sig_k in [-1,1]
        println("perturb the ", l0, "-th atom in direction ", i_l0,
                " with ", sig_l0, "and perturb the ", k,
                "-th atom in direction ", i_k, " with ", sig_k)
        X[i_l0,l0] += sig_l0 * h
        X[i_k,k] += sig_k * h
        atd = deepcopy(at)
        set_positions!(atd, X)
        X[i_l0,l0] -= sig_l0 * h
        X[i_k,k] -= sig_k * h
        # compute dEs on all neighbours as determined by data sketch
        for dat in data[Idk]
        if dat["i_l0"] == i_l0 && dat["i_k"] == i_k
            @assert dat["h"] == h  # test all h are the same!
            println("calculating forces ...")
            frc = forces(calc, atd)
            # write frc into the data point
            if !haskey(dat, key)
                dat[key] = zeros(size(frc))
            end
            dat[key] += frc * sig_l0*sig_k / (4.0*h^2)
        end
        end
    end
    end
end



# =============== General ROUTINES =============== #

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
   return (dVp - dVm) / (2*h)  # Vector{JVecF}
end

function _force_constants(calc, at::Atoms, l0, i, h)
    at[l0] += h * evec(i)
    Fp = forces(calc, at)
    at[l0] -= 2 * h * evec(i)
    Fm = forces(calc, at)
    at[l0] +=  h * evec(i)
    return (Fp - Fm) / (2*h)
    # return  mat( (Fp - Fm) / (2*h) )[:] |> collect
end
