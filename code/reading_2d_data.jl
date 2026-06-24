using JLD 
using LinearAlgebra

data_6 = load("data/data_6.jld")
data_12 = load("data/data_12.jld")
data_24 = load("data/data_24.jld")
data_48 = load("data/data_48.jld")
data_96 = load("data/data_96.jld")
data_192 = load("data/data_192.jld")
data_384 = load("data/data_384.jld")


#%%
#Determine the q-ratio 
q_u = zeros(5)
q_u[1] = abs(data_6["norm_u"][end]-data_12["norm_u"][end])/abs(data_12["norm_u"][end]-data_24["norm_u"][end])
q_u[2] = abs(data_12["norm_u"][end]-data_24["norm_u"][end])/abs(data_24["norm_u"][end]-data_48["norm_u"][end])
q_u[3] = abs(data_24["norm_u"][end]-data_48["norm_u"][end])/abs(data_48["norm_u"][end]-data_96["norm_u"][end])
q_u[4] = abs(data_48["norm_u"][end]-data_96["norm_u"][end])/abs(data_96["norm_u"][end]-data_192["norm_u"][end])
q_u[5] = abs(data_96["norm_u"][end]-data_192["norm_u"][end])/abs(data_192["norm_u"][end]-data_384["norm_u"][end])

q_p = zeros(5)
q_p[1] = abs(data_6["norm_p"][end]-data_12["norm_p"][end])/abs(data_12["norm_p"][end]-data_24["norm_p"][end])
q_p[2] = abs(data_12["norm_p"][end]-data_24["norm_p"][end])/abs(data_24["norm_p"][end]-data_48["norm_p"][end])
q_p[3] = abs(data_24["norm_p"][end]-data_48["norm_p"][end])/abs(data_48["norm_p"][end]-data_96["norm_p"][end])
q_p[4] = abs(data_48["norm_p"][end]-data_96["norm_p"][end])/abs(data_96["norm_p"][end]-data_192["norm_p"][end])
q_p[5] = abs(data_96["norm_p"][end]-data_192["norm_p"][end])/abs(data_192["norm_p"][end]-data_384["norm_p"][end])

#%%

CairoMakie.activate!()
set_theme!(theme_latexfonts())

Ns = [6,12,24,48,96]

fig = Figure(fontsize = 11,size = (700,300))

Label(fig[0,1:2],fontsize=14,latexstring("q\\text{-ratio for velocity and pressure when increasing } N"))

colgap!(fig.layout, 1, Relative(0.15))

ax1 = Axis(fig[1,1],
    title=latexstring("q-\\text{ratio for velocity}"),
    xlabel = L"N",
    ylabel = L"q",
    # xticks = [6,12,24,48],
    )
Makie.lines!(ax1,Ns,q_u,linewidth=2)
Makie.scatter!(ax1,Ns,q_u)

ax2 = Axis(fig[1,2],
    title=latexstring("q-\\text{ratio for pressure}"), 
    xlabel = L"N",
    ylabel = L"q",
    )
Makie.lines!(ax2,Ns,q_p,linewidth=2)
Makie.scatter!(ax2,Ns,q_p)

display(fig)
# save("figures/q-ratios.pdf", fig, px_per_unit = 1)



#%%
#Run the code for each N 5 times, then take the average 
#To compute runtime of time marching
times_N = zeros(7)
#N=6
times = [0.317736, 0.315429, 0.278292, 0.279990, 0.297386]
times_N[1] = mean(times)

#N=12
times = [0.899705, 0.819953,0.835814,1.189029,0.852107]
times_N[2] = mean(times)

#N=24
times = [  5.926382,5.845719 , 5.901325,5.713766 ,5.710743]
times_N[3] = mean(times)

#N=48
times = [47.079739 ,42.615171,42.765073,41.652889,42.655529]
times_N[4] = mean(times)

#N=96
times = [368.700576,352.520575,353.973150,354.548439,372.825337]
times_N[5] = mean(times)

#N=192 
#Takes about an hour each time 
times_N[6] = 3600

#N=384
#Predicted time from the progress bar of about 10 hours 
times_N[7] = 10*3600 

#Time to generate the mesh 
#N=6 to N=192, average already computed for smaller grids 
#N=384, only run once 
times_mesh = [0.00466,0.010402,0.060030,0.737373, 15.806628,273.026544,42000]

#%%

Ns = [6,12,24,48,96,192,384]

CairoMakie.activate!()
set_theme!(theme_latexfonts())


fig = Figure(fontsize = 11,size = (700,300))

Label(fig[0,1:2],fontsize=14,latexstring("\\text{Runtime to generate the grid and run the time marching over } N"))

colgap!(fig.layout, 1, Relative(0.15))

ax1 = Axis(fig[1,1],
    title=latexstring("\\text{Runtime to generate the grid}"),
    yscale=log10,
    xscale=log10,
    xlabel = L"N",
    ylabel = latexstring("\\text{Time}\\ (s)"),
    )
Makie.lines!(ax1,Ns,times_mesh,linewidth=2,label="Time")
Ns2 = [12,24,48,96,192]
Makie.lines!(ax1,Ns2,Ns2.^4*1e-8,linewidth=2,label=L"N^4")
Makie.scatter!(ax1,Ns,times_mesh)
axislegend(ax1,position = :rb)

ax2 = Axis(fig[1,2],
    title=latexstring("\\text{Runtime for time marching}"), 
    yscale=log10,
    xscale=log10,
    xlabel = L"N",
    ylabel = latexstring("\\text{Time}\\ (s)"),
    )
Makie.lines!(ax2,Ns,times_N,linewidth=2,label="Time")
Makie.lines!(ax2,Ns,Ns.^3*1e-4,linewidth=2,label=L"N^3")
Makie.scatter!(ax2,Ns,times_N)
axislegend(ax2,position = :rb)

display(fig)
save("figures/runtimes.pdf", fig, px_per_unit = 1)

#%%


N = 384


data = load("data/data_$(N).jld")

norm_u = data["norm_u"]
norm_p = data["norm_p"]
norm_div = data["norm_div"]

dx = 1/N 
if N == 384 
    dt = 1/(2*N)
else 
    dt = dx 
end

CairoMakie.activate!()
set_theme!(theme_latexfonts())

fig = Figure(fontsize = 16,size = (900, 700))

Label(fig[0,1:2],fontsize=22,latexstring("Re = $(Re),\\ N = $(N),\\ dt = $(round(dt,digits=5)),\\ dx = $(round(dx,digits=5))"))

colgap!(fig.layout, 1, Relative(0.05))
rowgap!(fig.layout,1,Relative(0.05))

if N != 384
    meshdata = load("data/meshdata_$(N).jld")
    mesh = meshdata["mesh"]
    points = meshdata["points"]
    ax1 = Axis(fig[1,1],title=latexstring("\\text{2D triangulated mesh with $(length(points)) vertices}"))
    Makie.wireframe!(ax1,mesh,transparency = true)
    Makie.scatter!(ax1,points)
end

ax2 = Axis(fig[1,2],
    title=latexstring("\\text{Velocity divergence norm}"), 
    # yscale=log10,
    xlabel = "Time iterates",
    ylabel = L"||Du_f||_2",
    )
Makie.lines!(ax2,norm_div,linewidth=2)

ax3 = Axis(fig[2,1],
    title=latexstring("\\text{Velocity norm}"),
    # yscale=log10,
    xlabel = "Time iterates",
    ylabel = L"||u_c-u_0||_2")
Makie.lines!(ax3,norm_u,linewidth=2)

ax4 = Axis(fig[2,2],
    title=latexstring("\\text{Pressure norm}"),
    # yscale=log10,
    xlabel = "Time iterates",
    ylabel = L"||p_c-p_0||_2")
Makie.lines!(ax4,norm_p,linewidth=2)

display(fig)
save("figures/N=$(N).pdf", fig, px_per_unit = 1)

#%% 
#Make plot of the convergence of ||p|| and ||u|| in space 
Ns = [6,12,24,48,96,192,384]

nus = zeros(7)
nps = zeros(7)

for i=1:7
    N = Ns[i]
    data = load("data/data_$(N).jld")
    nus[i] = data["norm_u"][end]
    nps[i] = data["norm_p"][end]
end

CairoMakie.activate!()
set_theme!(theme_latexfonts())

fig = Figure(fontsize = 11,size = (700,300))

Label(fig[0,1:2],fontsize=14,latexstring("\\text{Convergence of velocity and pressure norms over}\\ N"))

colgap!(fig.layout, 1, Relative(0.15))

ax1 = Axis(fig[1,1],
    title=latexstring("\\text{Velocity norm}"),
    # yscale=log10,
    # xscale=log10,
    xlabel = L"N",
    ylabel = L"||u_c-u_0||",
    )
Makie.lines!(ax1,Ns,nus,linewidth=2)
Makie.scatter!(ax1,Ns,nus)

ax2 = Axis(fig[1,2],
    title=latexstring("\\text{Pressure norm}"), 
    # yscale=log10,
    # xscale=log10,
    xlabel = L"N",
    ylabel = L"||p_c-p_0||",
    )
Makie.lines!(ax2,Ns,nps,linewidth=2)
Makie.scatter!(ax2,Ns,nps)

display(fig)
save("figures/spatial_convergence.pdf", fig, px_per_unit = 1)