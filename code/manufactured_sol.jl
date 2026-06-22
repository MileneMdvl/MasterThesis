"
This file contains the implementation of the manufactured solution, and particularly to determine the source term f that arises when substituting the manufactured solution into the dimensionless Navier-Stokes equations. 

This solution is only implemented in two dimensions. 

The solution u,v,p chosen here is such that it is divergence free and has Neumann boundary conditions on [0,1] x [0,√3/2]. The domain here is chosen so we can verify the implementation on the regular grid. 
"

using Symbolics
using Latexify

#Define x,y as variables 
@variables x, y 

#Define the Reynolds number
const Re = 1600

#Manufactured solution on [0,1] x [0,√3/2]
u = cos(2*pi*x) * sin(4*pi*y/sqrt(3))
v = - sin(2*pi*x) * cos(4*pi*y/sqrt(3))
p = cos(2*pi*x) * cos(4*pi*y/sqrt(3))

#Define functions to evaluate the manufactured solutions at discrete x,y values 
u_eval = eval(build_function(u,x,y))
v_eval = eval(build_function(v,x,y))
p_eval = eval(build_function(p,x,y))

#Define the differential operators 
Dx = Differential(x)
Dy = Differential(y)

#Compute the partial derivatives of velocity and the gradient of the pressure 
Du_Dx = expand_derivatives(Dx(u))
Du_Dxx = expand_derivatives(Dx(Du_Dx))

Du_Dy = expand_derivatives(Dy(u))
Dv_Dx = expand_derivatives(Dx(v))

Dv_Dy = expand_derivatives(Dy(v))
Dv_Dyy = expand_derivatives(Dy(Dv_Dy))

Dp_Dx = expand_derivatives(Dx(p))
Dp_Dy = expand_derivatives(Dy(p))

#Compute the resulting source term in both x and y directions 
f_x = eval(build_function(u * Du_Dx + v * Du_Dy + Dp_Dx - 1/Re * (Du_Dxx + Dv_Dyy), x, y))
f_y = eval(build_function(u * Dv_Dx + v * Dv_Dy + Dp_Dy - 1/Re * (Du_Dxx + Dv_Dyy), x, y))

