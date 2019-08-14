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
    # println("on reference: calculate Es")
    l0 = 1;
    ### Es = site_energy(tbm, atd, l0);
    # store Es
    dat = Dict{String, Any}();
    dat["Info"] = "Es of the origin site";
    ### dat["Es"] = Es;
    dat["datatype"] = "Es"
    push!(data, dat);

    # 1. compute dEs on reference and only store the derivatives
    # println("calculating dEs_", l0)
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
    for (n, neigs, R) in sites(nlist_d2)
        if n == l0
            index_rcut2_0 = neigs;
        end
    end
    N₂ = length(index_rcut2_0);
    d2Esh = zeros(3, N₂, 3, length(atd));
    dEs_pert_0p = zeros(3, N₂, 3, length(atd));
    dEs_pert_0n = zeros(3, N₂, 3, length(atd));
    for i = 1 : 3
        # println("perturb the origin atom in direction ", i, "+h :")
        # X[i,l0] += h;
        # atd = deepcopy(at);
        # set_positions!(atd, X);
        for j = 1 : N₂
            ℓ = index_rcut2_0[j];
            # println("calculating dE_", ℓ)
###         dEs_pert_0p[i, j, :, :] = site_energy_d(tbm, atd, ℓ) |> mat;
        end
        # println("perturb the origin atom in direction ", i, "-h :")
        # X[i,l0] -= 2*h;
        # atd = deepcopy(at);
        # set_positions!(atd, X);
        for j = 1 : N₂
            ℓ = index_rcut2_0[j];
            # println("calculating dE_", ℓ)
###         dEs_pert_0n[i, j, :, :] = site_energy_d(tbm, atd, ℓ) |> mat;
        end
        # X[i,l0] += h;
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
