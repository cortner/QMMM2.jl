using StaticArrays
using JuLIP, JuLIP.Potentials
using JuLIP.Potentials: site_energy, site_energy_d


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




# create the Dict for datas, but without any calculations
function data_djEs_sketch(at::Atoms, h::Float64, cutoffs::Array{Float64,1})

    # X  = positions(at) |> mat;
    atd = deepcopy(at);
    # neighbour lists for different cutoffs
    rcut1, rcut2, rcut3 = cutoffs[:];
    nlist_d1 = neighbourlist(at, rcut1);
    nlist_d2 = neighbourlist(at, rcut2);
    nlist_d3 = neighbourlist(at, rcut3);

    # calculators
    # eF = 10.89102; for Carbon and 5.521241 for Si
    # tbm = NRLTB.NRLTBModel(:C or Si,
    #                ZeroTGrand(eF = eF),
    #                bzquad = TB.GammaPoint(),
    #                cutoff = :original );

    # data
    D = Dict{String, Any}();
    D["species"] = JuLIP.Chemistry.chemical_symbol(at.Z[1]);
    D["at"] = Dict(atd);
    D["Info"] = "djEs type training set";
    D["cutoff_d"] = rcut1
    D["cutoff_d2"] = rcut2;
    D["cutoff_d3"] = rcut3;
    data = Dict[]
    D["data"] = data;

    # 0. compute Es on reference
    println("on reference: calculate Es")
    l0 = 1;
    ### Es = site_energy(tbm, atd, l0);
    # store Es
    dat = Dict{String, Any}();
    dat["Info"] = "Es of the origin site";
    ### dat["Es"] = Es;
    dat["datatype"] = "Es"
    push!(data, dat);

    # 1. compute dEs on reference and only store the derivatives
    println("calculating dEs_", l0)
    ### dEs_ref = site_energy_d(tbm, atd, l0) |> mat
    # store dEs
    dat = Dict{String, Any}();
    dat["Info"] = "dEs of the origin site";
### dat["dEs"] = dEs_ref;
    dat["datatype"] = "dEs"
    push!(data, dat);

    # 2. compute FD approximations for dE2 and only store the FD within rcut_d2
    #    d2Esh = 1/2h * ( E_{ℓ,n}(y+h⋅e0) - E_{ℓ,n}(y-h⋅e0) )
    index_rcut2_0 = [];
    for (n, neigs, r, R) in sites(nlist_d2)
        if n == l0
            index_rcut2_0 = neigs;
        end
    end
    N₂ = length(index_rcut2_0);
    d2Esh = zeros(3, N₂, 3, length(atd));
    dEs_pert_0p = zeros(3, N₂, 3, length(atd));
    dEs_pert_0n = zeros(3, N₂, 3, length(atd));
    for i = 1 : 3
        println("perturb the origin atom in direction ", i, "+h :")
        # X[i,l0] += h;
        # atd = deepcopy(at);
        # set_positions!(atd, X);
        for j = 1 : N₂
            ℓ = index_rcut2_0[j];
            println("calculating dE_", ℓ)
###         dEs_pert_0p[i, j, :, :] = site_energy_d(tbm, atd, ℓ) |> mat;
        end
        println("perturb the origin atom in direction ", i, "-h :")
        # X[i,l0] -= 2*h;
        # atd = deepcopy(at);
        # set_positions!(atd, X);
        for j = 1 : N₂
            ℓ = index_rcut2_0[j];
            println("calculating dE_", ℓ)
###         dEs_pert_0n[i, j, :, :] = site_energy_d(tbm, atd, ℓ) |> mat;
        end
        X[i,l0] += h;
        # compute the central finite differences
        for j = 1 : N₂
            ℓ = index_rcut2_0[j];
            index_rcut2_ℓ = [];
            # # TODO: make sure we really don't need this!
            # for (k, neigs, r, R) in sites(nlist_d2)
            #     if k == ℓ
            #         index_rcut2_ℓ = neigs;
            #     end
            # end
###         d2Esh[i, j, :, :] = 1.0/(2.0*h) *
                    # ( dEs_pert_0p[i,j,:,:] - dEs_pert_0n[i,j,:,:] );
            # store d2Esh
            dat = Dict{String, Any}();
            dat["Info"] = "perturb the origin atom in direction i
                and store d2Esh = 1/2h * ( E_{ℓ,n}(y+h*e₀) - E_{ℓ,n}(y-h*e₀) )";
            dat["h"] = h;
            dat["i"] = i;
            dat["l"] = ℓ;
            # dat["n"] = index_rcut2_ℓ;
###         dat["d2Esh"] = d2Esh[i, j, :, :];
            dat["datatype"] = "d2Esh";
            push!(data, dat);
        end
    end

    return D
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


eval_dat(::Val{:Es}, dat, calc, at) =
        site_energy(calc, at, 1)

eval_dat(::Val{:dEs}, dat, calc, at) =
        site_energy_d(calc, at, 1)

"""
TODO : document this datatype
"""
eval_dat(::Val{:d2Esh}, dat, calc, at) =
        _d2Esh(calc, at, 1, dat["l"], dat["i"], dat["h"])




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


function eval_dataset_tb!(::Val{d2Esh}, data, calc, at; key=key)
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
