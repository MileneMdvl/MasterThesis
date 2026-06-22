using JLD 
using LinearAlgebra

err_u = zeros(6)

data_6 = load("data/data_6.jld")
data_12 = load("data/data_12.jld")
data_24 = load("data/data_24.jld")
data_48 = load("data/data_48.jld")
data_96 = load("data/data_96.jld")
data_192 = load("data/data_192.jld")

#%%
#Determine the q-ratio 
q_u = zeros(4)
q_u[1] = abs(data_6["norm_u"][end]-data_12["norm_u"][end])/abs(data_12["norm_u"][end]-data_24["norm_u"][end])
q_u[2] = abs(data_12["norm_u"][end]-data_24["norm_u"][end])/abs(data_24["norm_u"][end]-data_48["norm_u"][end])
q_u[3] = abs(data_24["norm_u"][end]-data_48["norm_u"][end])/abs(data_48["norm_u"][end]-data_96["norm_u"][end])
q_u[4] = abs(data_48["norm_u"][end]-data_96["norm_u"][end])/abs(data_96["norm_u"][end]-data_192["norm_u"][end])

q_p = zeros(4)
q_p[1] = abs(data_6["norm_p"][end]-data_12["norm_p"][end])/abs(data_12["norm_p"][end]-data_24["norm_p"][end])
q_p[2] = abs(data_12["norm_p"][end]-data_24["norm_p"][end])/abs(data_24["norm_p"][end]-data_48["norm_p"][end])
q_p[3] = abs(data_24["norm_p"][end]-data_48["norm_p"][end])/abs(data_48["norm_p"][end]-data_96["norm_p"][end])
q_p[4] = abs(data_48["norm_p"][end]-data_96["norm_p"][end])/abs(data_96["norm_p"][end]-data_192["norm_p"][end])

#%%

CairoMakie.activate!()
set_theme!(theme_latexfonts())


fig = Figure(fontsize = 11,size = (600,300))

ax1 = Axis(fig[1,1],
    title=latexstring("q-\\text{ratio for velocity}"),
    xlabel = "N",
    ylabel = L"q",
    # xticks = [6,12,24,48],
    )
Makie.lines!(ax1,[6,12,24,48],q_u,linewidth=2)
Makie.scatter!(ax1,[6,12,24,48],q_u)

ax2 = Axis(fig[1,2],
    title=latexstring("q-\\text{ratio for pressure}"), 
    xlabel = "N",
    ylabel = L"q",
    )
Makie.lines!(ax2,[6,12,24,48],q_p,linewidth=2)
Makie.scatter!(ax2,[6,12,24,48],q_p)

display(fig)
save("figures/q-ratios.pdf", fig, px_per_unit = 1)



#%%
#Run the code for each N 5 times, then take the average 
#To compute runtime
times_N = zeros(6)
#N=6
times = [2.913592,2.858054,2.789177,2.801208,2.880923]
time_N[1] = mean(times)

#N=12
times = [3.493754,4.170753, 3.135380,3.051201,3.085528]
times_N[2] = mean(times)

#N=24
times = [ 8.040673,7.726412,7.619538, 7.734748,7.647911 ]
times_N[3] = mean(times)

#N=48
times = [46.611465,46.893828,45.181199,47.975820,45.399184]
times_N[4] = mean(times)

#N=96