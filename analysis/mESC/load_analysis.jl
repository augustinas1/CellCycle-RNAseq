println(@__DIR__)
srcpath = normpath(joinpath(@__DIR__, "../../src/"))
include(srcpath*"dists.jl")
include(srcpath*"mle.jl")
include(srcpath*"confidence.jl")

include("load_merged_data.jl")

# load cell-cycle phase dependent fits
fitpath = normpath(datapath*"fits_phase/")
@load fitpath*"phase_th_dep_CI_results.jld2" th_dep_res
@load fitpath*"phase_th_ind_CI_results.jld2" th_ind_res

G1_th_dep_fits = th_dep_res["G1_fits"]
G2M_th_dep_fits = th_dep_res["G2M_fits"]
G1_th_dep_CIs = th_dep_res["G1_CIs"]
G2M_th_dep_CIs = th_dep_res["G2M_CIs"]
gene_inds_th_dep = th_dep_res["gene_inds"]

G1_th_ind_fits = th_ind_res["G1_fits"]
G2M_th_ind_fits = th_ind_res["G2M_fits"]
G1_th_ind_CIs = th_ind_res["G1_CIs"]
G2M_th_ind_CIs = th_ind_res["G2M_CIs"]
gene_inds_th_ind = th_ind_res["gene_inds"]

# load merged data fits
fitpath = normpath(datapath*"fits_merged/")
@load fitpath*"merged_th_ind_CI_results.jld2" th_ind_res
@load fitpath*"merged_th_dep_CI_results.jld2" th_dep_res

merged_th_ind_fits = th_ind_res["fits"]
merged_th_ind_CIs = th_ind_res["CIs"]
gene_inds_merged_th_ind = th_ind_res["gene_inds"]

merged_th_dep_fits = th_dep_res["fits"]
gene_inds_merged_th_dep = th_dep_res["gene_inds"]

shared_gene_inds = intersect(gene_inds_th_dep, gene_inds_th_ind, gene_inds_merged_th_ind, gene_inds_merged_th_dep)
println("Shared genes: ", length(shared_gene_inds))
gene_names = gene_names[shared_gene_inds]

# suitable genes are only those which in both phases are fit by NegativeBinomial or more complex distributions
# we next consider only the suitable genes shared between all datasets
# e.g. gene X is not suitable if in the merged dataset it is best fit by BetaPoisson but in the G1 dataset by Poisson 
th_dep_inds = [findfirst(gene_inds_th_dep.==ind) for ind in shared_gene_inds]
th_ind_inds = [findfirst(gene_inds_th_ind.==ind) for ind in shared_gene_inds]
th_ind_merged_inds = [findfirst(gene_inds_merged_th_ind.==ind) for ind in shared_gene_inds]
th_dep_merged_inds = [findfirst(gene_inds_merged_th_dep.==ind) for ind in shared_gene_inds]

G1_th_dep_fits = G1_th_dep_fits[th_dep_inds]
G2M_th_dep_fits = G2M_th_dep_fits[th_dep_inds]
G1_th_ind_fits = G1_th_ind_fits[th_ind_inds]
G2M_th_ind_fits = G2M_th_ind_fits[th_ind_inds]
merged_th_dep_fits = merged_th_dep_fits[th_dep_merged_inds]
merged_th_ind_fits = merged_th_ind_fits[th_ind_merged_inds]

G1_th_dep_CIs = G1_th_dep_CIs[th_dep_inds]
G2M_th_dep_CIs = G2M_th_dep_CIs[th_dep_inds]
G1_th_ind_CIs = G1_th_ind_CIs[th_ind_inds]
G2M_th_ind_CIs = G2M_th_ind_CIs[th_ind_inds]

xG1 = xG1[shared_gene_inds]
xG2M = xG2M[shared_gene_inds]
xS = xS[shared_gene_inds]
counts_merged = counts_merged[shared_gene_inds]
counts_spliced = counts_spliced[shared_gene_inds]
ngenes = length(shared_gene_inds)

# Figure settings
using CairoMakie

fontloc_regular = "/usr/share/fonts/truetype/msttcorefonts/Arial.ttf"
fontloc_bold = "/usr/share/fonts/truetype/msttcorefonts/Arial_Bold.ttf"
fontloc_bold_italic = "/usr/share/fonts/truetype/msttcorefonts/Arial_Bold_Italic.ttf"
fontloc_italic = "/usr/share/fonts/truetype/msttcorefonts/Arial_Italic.ttf"

fig_theme = Theme(fonts=(regular = fontloc_regular, bold=fontloc_bold, 
                        bold_italic=fontloc_bold_italic, italic=fontloc_italic),
                  fontsize=7,
                  Axis=(xgridvisible=false, ygridvisible=false, 
                       xticksize=1.5, yticksize=1.5,
                       xticklabelpad=1, xlabelpadding=0.5,
                       yticklabelpad=1, ylabelpadding=2,
                       xtickwidth=0.7,
                       ytickwidth=0.7,
                       spinewidth=0.7),
                 linewidth=0.7)
set_theme!(fig_theme)
CairoMakie.activate!(type = "svg")

size_inches = (2.23, 1.5)
size_pt = 72 .* size_inches

c1 = colorant"#0098d1"
c2 = colorant"#f07269"
c3 = colorant"#e15165ff";