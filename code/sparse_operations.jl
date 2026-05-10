#This file includes the operations for sparse matrices, namely: 
# - Evaluate
# - SparseInnerProduct
# - SparseMatVec 

include("mesh_functions.jl")

#Function: Evaluate
#Input: Tensor G, Array A (either face or cell)
#Output: G[A], the value of G on A 
function Evaluate(G,A)
    ind_A = CartesianIndex(Tuple(A))
    return G[ind_A]
end

#Function SparseInnerProduct
#Input A,B: sparse arrays to compute the inner product of
#Output A⋅B Float: Inner Product (A,B) 
function SparseInnerProduct(A,B, type::String)
    if size(A) != size(B)
        println("Error: dimensions of input do not match")
        println("-> ",size(A), " does not equal ", size(B))
        return 
    end

    if size(A)[1] != size(A)[2]
        println("Error: input not square")
        println("-> ",size(A)[1], " does not equal ", size(A)[2])
        return 
    end

    #For the face innerproduct, we only want to sum over edge (i,j) not also
    #(j,i) (same for 3d)

    if type == "cell"
        indices_to_sum = UniqueList(cell_list)
    elseif  type == "face"
        indices_to_sum = UniqueList(face_list)
    end

    innerprod = 0

    for i in eachindex(A)
            ind = collect(Tuple.(i))
            if ind in indices_to_sum 
                if hasindex(A,i) && hasindex(B,i)
                    entry = A[i] * B[i] * Volume(ind) 
                    if type == "face"
                        entry *= DualEdge(ind)
                    end
                    innerprod += entry
                end
            end
        end

    if type != "cell" && type != "face"
        println("Error: type must be either 'cell' or 'face'")
    end
    
    return innerprod                
end

#Function InnerProdCell 
#Input: a, b Vector defined on cell centers 
#       C Vector of size nc: list of cells 
#Output: Discrete inner product a⋅b 
function InnerProdCell(a,b,C)
    ip = 0
    nc = length(C)
    if length(a) == length(b) == nc 
        for i in 1:nc 
            ip += Volume(C[i]) * a[i] * b[i]
        end
    else
        println("Error: incorrect vector lengths, should correspond to the number of cells, ", nc)
        printltn("Otherwise try 'InnerProdFace'")
    end
    return ip
end

#Function InnerProdFace 
#Input: a, b Vector defined on face centers 
#       F Vector of size nf: list of (unique) faces 
#Output: Discrete inner product a⋅b 
function InnerProdFace(a,b,F)
    ip = 0
    nf = length(F)
    if length(a) == length(b) == nf
        for i in 1:nf 
            e = F[i]
            ip += Volume(e) * DualEdge(e) * a[i] * b[i]
        end
    else
        println("Error: incorrect vector lengths, should correspond to the number of unique faces, ", nf)
        println("Otherwise try 'InnerProdCell'")
    end
    return ip
end


#Function SparseMatVec
#Input: A Sparse Matrix 
#       b Sparse/Dense Vector 
#Output: Ab Vector, result of A*b
function SparseMatVec(A,b)
    Ab = zeros(size(A)[1])
    #Get indices in which A is nonzero
    ind_nzA = findnz(A)[1]
    #Store b as a sparse vector 
    bb = NDSparseArray(b)
    ind_nzb = findnz(bb)[1] 
    for ij in ind_nzA
        i,j = Tuple(ij)
        j = CartesianIndex(j)
        if j in ind_nzb
            Ab[i] += A[ij] * b[j]
        end
    end
    return Ab
end

#Function SparseVecMat
#Input: b Dense Vector
#       A Sparse Matrix  
#Output: bA Vector, result of b*A
function SparseVecMat(b,A)
    bA = zeros(size(b))
    #Get indices in which A is nonzero
    ind_nzA = findnz(A)[1]
    #Store b as a sparse array for the sake of dimensions 
    bb = NDSparseArray(b)
    ind_nzb = findnz(bb)[1] 
    for ij in ind_nzA
        i,j = Tuple(ij)
        if i in ind_nzb
            bA[j] += A[ij] * b[i]
        end
    end
    return bA
end
