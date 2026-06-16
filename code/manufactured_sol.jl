using Symbolics
using Latexify

@variables x, y 

# @variables L, nu, pi
# @variables rho_0, rho_x, rho_y, rho_xy
# @variables u_0, u_x, u_y, u_xy
# @variables v_0, v_x, v_y, v_xy
# @variables p_0, p_x, p_y, p_xy

# @variables a_rhox, a_rhoy, a_rhoxy
# @variables a_ux, a_uy, a_uxy
# @variables a_vx, a_vy, a_vxy
# @variables a_px, a_py, a_pxy

# const l = 1
# const mu = 10

# const rho_0 = 1 
# const rho_x = 0.1 
# const rho_y = 0.15 
# const rho_xy = 0.08 
# const a_rhox = 0.75 
# const a_rhoy = 1.0
# const a_rhoxy = 1.25 

# const u_0 = 70 
# const u_x = 4
# const u_y = -12 
# const u_xy = 7
# const a_ux = 5/3
# const a_uy = 1.5
# const a_uxy = 0.6

# const v_0 = 90
# const v_x = -20 
# const v_y = 4
# const v_xy = -11 
# const a_vx = 1.5
# const a_vy = 1.0 
# const a_vxy = 0.9 

# const p_0 = 1e5
# const p_x = -0.3e5
# const p_y = 0.2e5
# const p_xy = -0.25e5 
# const a_px = 1.0 
# const a_py = 1.25
# const a_pxy = 0.75

# rho = rho_0 + rho_x * sin(a_rhox*pi*x/l) + rho_y * cos(a_rhoy*pi*y/l) + rho_xy * cos(a_rhoxy*pi*x*y/l)
# u = u_0 + u_x * sin(a_ux*pi*x/l) + u_y * cos(a_uy*pi*y/l) + u_xy * cos(a_uxy*pi*x*y/l)
# v = v_0 + v_x * sin(a_vx*pi*x/l) + v_y * cos(a_vy*pi*y/l) + v_xy * cos(a_vxy*pi*x*y/l)
# p = p_0 + p_x * sin(a_px*pi*x/l) + p_y * cos(a_py*pi*y/l) + p_xy * cos(a_pxy*pi*x*y/l)

const Re = 1600

#Manufactured solution on [0,1] x [0,√3/2]
u = cos(2*pi*x) * sin(4*pi*y/sqrt(3))
v = - sin(2*pi*x) * cos(4*pi*y/sqrt(3))
# u = 1
# v = 0
p = cos(2*pi*x) * cos(2*pi*y)

# rho_eval = eval(build_function(rho,x,y))
u_eval = eval(build_function(u,x,y))
v_eval = eval(build_function(v,x,y))
p_eval = eval(build_function(p,x,y))

Dx = Differential(x)
Dy = Differential(y)

Du_Dx = expand_derivatives(Dx(u))
Du_Dxx = expand_derivatives(Dx(Du_Dx))

Du_Dy = expand_derivatives(Dy(u))
Dv_Dx = expand_derivatives(Dx(v))

Dv_Dy = expand_derivatives(Dy(v))
Dv_Dyy = expand_derivatives(Dy(Dv_Dy))

Dp_Dx = expand_derivatives(Dx(p))
Dp_Dy = expand_derivatives(Dy(p))


f_x = eval(build_function(u * Du_Dx + v * Du_Dy + Dp_Dx - 1/Re * (Du_Dxx + Dv_Dyy), x, y))
f_y = eval(build_function(u * Dv_Dx + v * Dv_Dy + Dp_Dy - 1/Re * (Du_Dxx + Dv_Dyy), x, y))

