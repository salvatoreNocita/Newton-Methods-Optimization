import numpy as np
import scipy as sci
import scipy.sparse as scis

class Solvers(object):
    """ This class exploits Cholensky methods to support algorithms
    """

    def __init__(self):
        pass

    def Build_bk(self,hessf: np.array,k_max: int,corr_fact: float) -> np.ndarray:
        beta = 1e-3
        try:
            import scipy.sparse as sp
            if sp.issparse(hessf):
                diag_elements = hessf.diagonal()
                H = hessf.toarray()
            else:
                diag_elements = np.diag(hessf)
                H = hessf
        except Exception:
            diag_elements = np.diag(hessf)
            H = hessf
        tau0 = 0 if np.min(diag_elements) > 0 else (-np.min(diag_elements) + beta)
        tauk = tau0
        bk = H + tauk * np.identity(H.shape[0])
        flag = False
        i = 0
        while not flag and i < k_max:
            tauk_1 = max(corr_fact * tauk, beta)
            bk = bk + tauk_1 * np.identity(bk.shape[0])
            tauk = tauk_1
            try:
                L = np.linalg.cholesky(bk)
                flag = True
                return L, bk
            except np.linalg.LinAlgError:
                flag = False
            i += 1
        raise np.linalg.LinAlgError(f"Hessian can't be modified with {k_max}")

    def ichol(self,A,drop_tol : float = 1e-4):
        """ This function performs Incomplite Cholesky Factorization, i.e. computes a lower triangular matrix L
            such that A=LL^T, without computing all elements as Cholesky (esxploits sparsity of the matrix A) 
            INPUT:
            - A : np.ndarray -> Sparse and SPD matrix
            - drop_tol: float -> discard L[j,i] if |L[j,i]| <= drop_tol
            OUTPUT:
            - L : np.ndarray -> Lower triangular matrix 
        """
        if not scis.isspmatrix(A):                                   #Check ifa matrix is sparse
            A = scis.csc_matrix(A)
        elif not scis.isspmatrix_csc(A):
            A = A.tocsc()                                           #We want to work in csc format

        n = A.shape[0]
        L = scis.lil_matrix((n,n))                                       #initialize empty matrix
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
        if scis.isspmatrix(L):
            # Sparse triangular solves: preserve sparsity
            y = sci.sparse.linalg.spsolve_triangular(L, -gradf, lower=True)
            x = sci.sparse.linalg.spsolve_triangular(L.T, y, lower=False)
            return x.flatten()
        else:
            # Dense Cholesky factor: use NumPy solves
            y = np.linalg.solve(L, -gradf)
            x = np.linalg.solve(L.T, y)
            return x.flatten()
    
    def CG_Find_pk(self,bk:np.array,gradf: np.array,precond: str='yes') -> np.array:
        if precond == 'yes':
            L = self.ichol(bk,drop_tol=1e-4)

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

    def make_symmetric(self, Hessf):
        if scis.issparse(Hessf):
            diff = (Hessf - Hessf.T).tocoo()                                    #(COO) format, (row_idx,col_idx,data)
            if diff.data.size > 0 and not np.all(np.abs(diff.data) <= 1e-8):    #data.size = how many non null elements
                return 0.5 * (Hessf + Hessf.T)
            else:
                return Hessf
        else:
            if not np.allclose(Hessf, Hessf.T, atol=1e-8):
                return 0.5 * (Hessf + Hessf.T)
            else:
                return Hessf