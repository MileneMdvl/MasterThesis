#This file contains all the functions needed to build the stiffness matrix and
#load vector to solve the Poisson problem for pressure 

#To do: 
#      - Clean up file 
#      - Implement test case 


#Generalise the function to vectorise an array depending on the dimension 
#Function Vectorise
#Input: a, array to vectorise 
#       type, either cell or face, depending on if a is an array over cells
#       (e.g. p) or an array over faces (e.g. u)
#Output: a_vec, vector of a defined on the vertices of the discretisation
function Vectorise(a,type)
    np = length(vertex_list)
    a_vec = zeros(np)
    for i in 1:np 
        if type == "cell"
            n_cells = length(cell_list)
                num = 0
                denom = 0
                for k in 1:n_cells 
                    K=cell_list[k]
                    if i in K 
                        num += a[K[1],K[2],K[3]] * Volume(K) 
                        denom += Volume(K) 
                    end
                end
        elseif type == "face"
            n_faces = length(face_list)
            for i in 1:np 
                num = 0
                denom = 0
                for j in 1:n_faces
                    e=face_list[j]
                    if i in e 
                        num += a[e[1],e[2]] * Volume(e) 
                        denom += Volume(e) 
                    end
                end
            end
        else 
            println("Error: incorrect type, should be either 'cell' or 'face'")
        end
        a_vec[i] = num/denom 
    end
    return a_vec 
end

#Build the stiffness matrix A
function StiffnessMatrix(np)
    stiff_mat = SparseArrays.spzeros(np,np)
    for i in 1:np 
        for j in 1:np 
            for k in 1:n_cells 
                K = cell_list[k]
                if i in K && j in K 
                    index_i = findall(x->x==i,K)[1]
                    index_j = findall(x->x==j,K)[1]
                    stiff_mat[i,j] += BuildStiffElem(index_i,index_j,K)
                end
            end
        end
    end
    return stiff_mat
end

#Define the Jacobian determinant 
function JacobianDeterminant(K)
    p = zeros(3,2)
    for i in 1:3 
        p[i,:] = vertex_list[K[i]]
    end
    det = (p[1,1]-p[3,1])*(p[2,2]-p[3,2])-(p[2,1]-p[3,1])*(p[1,2]-p[3,2])
    return det 
end

#Define the partial derivatives
function dxdxi_1(K) 
    return vertex_list[K[1]][1] - vertex_list[K[3]][1]
end

function dydxi_1(K) 
    return vertex_list[K[1]][2] - vertex_list[K[3]][2]
end

function dxdxi_2(K) 
    return vertex_list[K[2]][1] - vertex_list[K[3]][1]
end

function dydxi_2(K) 
    return vertex_list[K[2]][2] - vertex_list[K[3]][2]
end

#∂ϕᵢ / ∂ξ₁
function dphi_dxi_1(i)
    if i == 1
        return 1
    elseif i == 2
        return 0 
    elseif i == 3 
        return -1 
    end
end

#∂ϕᵢ / ∂ξ₂
function dphi_dxi_2(i)
    if i == 1
        return 0
    elseif i == 2
        return 1 
    elseif i == 3 
        return -1 
    end
end

#Build the stiffness element for entry i,j belonging to cell K 
function BuildStiffElem(i,j,K)
    grad_phi_i = [dphi_dxi_1(i),dphi_dxi_2(i)]
    grad_phi_j = [dphi_dxi_1(j),dphi_dxi_2(j)]
    M = [dydxi_2(K) -dydxi_1(K); -dxdxi_2(K) dxdxi_1(K)] 
    det = JacobianDeterminant(K) 
    A_ij = 1/2 * 1/det * dot(M * grad_phi_i, M * grad_phi_j)
    return A_ij
end



#Build the load vector f 
load_vec = zeros(np)

function Advection(u)
    u_vec = Vectorise(u,"face")
    np = length(u)
    ∇u = zeros(np,np)
    for i in 1:np
        for j in 1:np
            if [i,j] ∈ face_list
                ∇u = Gradient()
            end
        end
    end
end

#Build the load element for vertex i belonging to cell K
function BuildLoadElem(i,K)
    return 1/2 
end

#Get the value (average) of u on a cell K 
function CellAverage(u,K)
    num = 0
    denom = 0
    e_K = Faces(K)
    for i in eachindex(K)
        e = e_K[i,:]
        num += u[e[1],e[2]] * FaceSurface(e) 
        denom += FaceSurface(e) 
    end    
    return num/denom 
end
