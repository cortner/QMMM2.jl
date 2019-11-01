# create the Dict for datas, but without any calculations
function data_djEs_sketch(at::Atoms, h::Float64, cutoffs::Array{Float64,1})

    atd = deepcopy(at);
    # neighbour lists for different cutoffs
    rcut1, rcut2, rcut3 = cutoffs[:];
    nlist_d1 = neighbourlist(at, rcut1);
    nlist_d2 = neighbourlist(at, rcut2);
    nlist_d3 = neighbourlist(at, rcut3);

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

    # 0. Es
    l0 = 1;
    dat = Dict{String, Any}();
    dat["Info"] = "Es of the origin site";
    dat["datatype"] = "Es"
    push!(data, dat);

    # 1. dEs
    dat = Dict{String, Any}();
    dat["Info"] = "dEs of the origin site";
    dat["datatype"] = "dEs"
    push!(data, dat);

    # 2. d2Es
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
        for j = 1 : N₂
            ℓ = index_rcut2_0[j];
            index_rcut2_ℓ = [];
            # store d2Esh
            dat = Dict{String, Any}();
            dat["Info"] = "perturb the origin atom in direction i
                and store d2Esh = 1/2h * ( E_{ℓ,n}(y+h*e₀) - E_{ℓ,n}(y-h*e₀) )";
            dat["h"] = h;
            dat["i"] = i;
            dat["l"] = ℓ;
            dat["datatype"] = "d2Esh";
            push!(data, dat);
        end
    end

    return D
end
