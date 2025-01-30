dirpath = normpath(joinpath(@__DIR__, "../../."))
using Pkg; Pkg.activate(dirpath*"/.")

srcpath = dirpath*"src/age-independent/"
include(srcpath*"dists.jl")
include(srcpath*"mle.jl")
include(srcpath*"confidence.jl")
include(dirpath*"analysis/mESC/load_data.jl")

# --- Filter by fit complexity ---

# load cell-cycle phase dependent fits
fitpath = normpath(datapath*"fits_age-independent/")
@load fitpath*"th_ind_CI_results.jld2" th_ind_res

G1_th_ind_fits = th_ind_res["G1_fits"]
G2M_th_ind_fits = th_ind_res["G2M_fits"]
#G1_th_ind_CIs = th_ind_res["G1_CIs"]
#G2M_th_ind_CIs = th_ind_res["G2M_CIs"]
gene_inds = th_ind_res["gene_inds"]

# suitable genes are only those which in both phases are fit by NegativeBinomial or more complex distributions
println("Number of bursty genes: ", length(gene_inds))
gene_names = gene_names[gene_inds]
xG1 = xG1[gene_inds]
xG2M = xG2M[gene_inds]
xS = xS[gene_inds]
counts_spliced = counts_spliced[gene_inds]
ngenes = length(gene_inds)

# --- Cell age computations ---

# find θ values corresponding to cell cycle phase bounds and transitions
θ_G1_f = maximum(thetaG1)
θ_S_i = minimum(thetaS)
θ_S_f = maximum(thetaS)
θ_G2M_i = minimum(thetaG2M)
θ_G1_S = θ_G1_f + (θ_S_i - θ_G1_f)/2
θ_S_G2M = θ_S_f + (θ_G2M_i - θ_S_f)/2

# find unique θ values and their distribution in each cell cycle phase
θs_G1 = sort(unique(thetaG1))
θs_S = sort(unique(thetaS))
θs_G2M = sort(unique(thetaG2M))
θs = vcat(θs_G1, θs_S, θs_G2M)
n_θs = zeros(Int, length(θs))

for i in eachindex(θs)
    θ = θs[i]
    n = count(θ .== theta)
    n_θs[i] = n
end

inds_θs_G1 = [findall(th .== thetaG1) for th in θs_G1]
inds_θs_S = [findall(th .== thetaS) for th in θs_S]
inds_θs_G2M = [findall(th .== thetaG2M) for th in θs_G2M]

# number of cells in each cell cycle phase
nG1 = length(thetaG1)
nS = length(thetaS)
nG2M = length(thetaG2M)
n_cells = nG1 + nS + nG2M

# --- The median cell cycle length for mESCs in 2i + LIF

T_cycle = 13.25 #h

# --- Figure settings ---

using CairoMakie

#fontloc_regular = "/usr/share/fonts/truetype/msttcorefonts/Arial.ttf"
#fontloc_bold = "/usr/share/fonts/truetype/msttcorefonts/Arial_Bold.ttf"
#fontloc_bold_italic = "/usr/share/fonts/truetype/msttcorefonts/Arial_Bold_Italic.ttf"
#fontloc_italic = "/usr/share/fonts/truetype/msttcorefonts/Arial_Italic.ttf"
fontloc_regular = "Arial"
fontloc_bold = "Arial Bold"
fontloc_bold_italic = "Arial Bold Italic"
fontloc_italic = "Arial Italic"

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
c3 = colorant"#e15165ff"
cx1 = colorant"#0098D1"
cx2 = colorant"#8790fd"
cx3 = colorant"#f07269"


# Increase Makie violing npoints to make the kernel density plots smoother
# NOTE: this may not be working as intended after package update
import CairoMakie: violin, violin!
import Makie: automatic

@recipe Violin (x, y) begin
    npoints = 200
    boundary = automatic
    bandwidth = automatic
    "vector of statistical weights (length of data). By default, each observation has weight `1`."
    weights = automatic
    "Specify `:left` or `:right` to only plot the violin on one side."
    side = :both
    "Scale density by area (`:area`), count (`:count`), or width (`:width`)."
    scale = :area
    "Orientation of the violins (`:vertical` or `:horizontal`)"
    orientation = :vertical
    "Width of the box before shrinking."
    width = automatic
    dodge = automatic
    n_dodge = automatic
    "Shrinking factor, `width -> width * (1 - gap)`."
    gap = 0.2
    dodge_gap = 0.03
    "Specify values to trim the `violin`. Can be a `Tuple` or a `Function` (e.g. `datalimits=extrema`)."
    datalimits = (-Inf, Inf)
    max_density = automatic
    "Show median as midline."
    show_median = false
    mediancolor = @inherit linecolor
    medianlinewidth = @inherit linewidth
    color = @inherit patchcolor
    strokecolor = @inherit patchstrokecolor
    strokewidth = @inherit patchstrokewidth
    Makie.MakieCore.mixin_generic_plot_attributes()...
    cycle = [:color => :patchcolor]
end