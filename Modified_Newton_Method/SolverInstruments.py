import numpy as np
import scipy as sci
from scipy.sparse import lil_matrix

class Solvers(object):
    """ This class exploits Cholensky methods to support algorithms
    """

    def __init__(self):
        pass

    def ichol(self,A,drop_tol : float = 1e-4):
        """ This function performs Incomplite Cholesky Factorization, i.e. computes a lower triangular matrix L
            such that A=LL^T, without computing all elements as Cholesky (esxploits sparsity of the matrix A) 
            INPUT:
            - A : np.ndarray -> Sparse and SPD matrix
            - drop_tol: float -> discard L[j,i] if |L[j,i]| <= drop_tol
            OUTPUT:
            - L : np.ndarray -> Lower triangular matrix 
        """
        if not sci.isspmatrix(A):                                   #Check ifa matrix is sparse
            A = sci.csc_matrix(A)
        elif not sci.isspmatrix_csc(A):
            A = A.tocsc()                                           #We want to work in csc format

        n = A.shape[0]
        L = lil_matrix((n,n))                                       #initialize empty matrix
        eps = 1e-15                                                 #Tollerance

        for i in range(n):
            diag = A[i,i] - (L[i,:i].power(2).sum())                #elements among diagonal
            if diag <= 0:
                diag = max(diag,eps)                                #Avoid negativeness coming from approximations
            L[i,i] = np.sqrt(diag)                                  #Put diagonal on L
            
            #Store elements used to compute products next
            #(Out of loop, compute only ones)
            Li = L[i,:i].toarray().ravel()          
            Lii = L[i,i]

            #csc format: data = non_zero elements, 
            #indices = position (reading by column) of non zero element
            col_i = A.getcol(i)
            data = col_i.data 
            rows = col_i.indices

            for r,j in enumerate(rows):                     #Same as classic Cholensky but only if non zero element
                if j <= i:
                    continue
                aji = data[r]
                Lj = L[j,:i].toarray().ravel()
                val = (aji - np.dot(Lj,Li))/Lii

                if drop_tol > 0.0 and abs(val) <= drop_tol:
                    continue
                L[j,i] = val

        return L.tocsr()
    
    def chol_Find_Pk(self,L: np.array,gradf: np.array) -> np.array:
        forward_solving= sci.sparse.linalg.solve(L,-gradf)
        backward_solving= sci.sparse.linalg.solve(L.T,forward_solving)
        return backward_solving.flatten()
    
    def CG_Find_pk(self,bk:np.array,gradf: np.array) -> np.array:
        if self.precond == 'yes':
            L = Solvers.ichol(bk,drop_tol=1e-4)

            #We want to avoid to compute and store M=LL^-1 so We create a function that provides to 
            # the Linear operator istruction to how M^-1 (used for CG) shoul works)
            def precond_solver(v: np.array):
                """This function compute Mx = v avoiding explicit constraction of preconditioner M. 
                Linear Operator is defining the action of M^-1 on a vector within matvec """
                y = sci.sparse.linalg.spsolve_triangular(L, v, lower=True)  # Risolvi L y = v
                x = sci.sparse.linalg.spsolve_triangular(L.T, y, lower=False)  # Risolvi L^T x = y
                return x
            
            M_inv = sci.sparse.linalg.LinearOperator(shape = bk.shape, matvec = precond_solver)
            pk, info= sci.sparse.linalg.cg(bk,-gradf, M= M_inv)
        else:
            pk, info= sci.sparse.linalg.cg(bk,-gradf)

        if info != 0:
            print("CG do not converge")

        return pk
    
    def make_symmetric(self,Hessf: np.array,xk: np.array):
        hessf= Hessf.copy()
        if np.allclose(hessf,hessf.T, atol= 10e-8) == False:
            return 0.5 * (hessf + hessf.T)
        else:
            return hessf