import numpy as np
import scipy as sci
from scipy.sparse import lil_matrix
from Project.model.Derivatives import ApproximateDerivatives
from Project.model.Derivatives import ExactDerivatives
from Project.model.Derivatives import SparseApproximativeDerivatives
from Project.model.Functions import FunctionDefinition

class ModifiedNewton(object):
    """Il Newton method è un algoritmo iterativo che consente di sfruttare la parsità matriciale per convergere ad una soluzione che minimizza la funzione obiettivo.
       Il tasso di convergenza è quadratico quindi più dispendioso dello SteepestDescent, essendo che sfrutta le informazioni date dalla derivata seconda. Inoltre, la
       direzione di movimento ad ogni passo dell'algoritmo, viene individuata risolvendo un sistema lineare. Il modified Newton Method risolve il problema del mancato
       funzionamento del metodo nell'evenienza in cui la matrice hessiana è non positive definite.
       Attributi:
       - x0: np.array= vettore colonna che rappresenta il punto di partenza del metodo
       - alpha0: int= 1, scelta obbligata nel Newton Method per garantire la convergenza quadratica
       - f: callable Rn --> R = funzione obiettivo da minimizzare
       - gradf: callable Rn --> Rn = gradiente della funzione valutata in x
       - Hessf: callable Rn --> R(nxn) = matrice hessiana, contiene le derivate seconde
       - kmax: int = massimo numero di iterazioni del metodo
       - tolgrad: float = tolleranza della norma del gradiente, usata come stopping criterion
       - c1: float = fattore per definire la validicità delle Armijo conditions nella scelta di alphak
       - rho: float = fattore per la scelta di alpha k con strategia di backtracking
       - btmax: int = massimo numero di iterazioni della backtracking
       - eps: float = precisione macchina
       - solver_linear_system: string = metodo di risoluzione del sistema lineare per trovare la direzione di movimento 
                               del passo dell'algoritmo.
       - H_correction_factor: float = fattore con cui verrà corretta la matrice hessiana se non è positive definite
       - Derivatives: str = Determina se calcolare le derivate esarre di gradiente ed hessian o se usare finite differences
       - Derivative_method: str = forward, backward or central derivative method for finite differences
       - Perturbation: float = value of h to compute approximation of derivatives
       Variabili:
       - k: int= contatore delle uiterazioni del metodo
       - bt: int= contatore delle iterazioni della backtracking
       - x_seq: list(np.array)= sequenza delle soluzioni trovate dal metodo
       - bt_seq: list(int)= sequenza dei numeri di iterazioni di backtracking usate ad ogni iterazione del metodo, per la scelta di alphak
       - xk: np.array= vettore soluzionme dell'iterazione k del Newton Method
       - alphak: float= step-length usato per il passo k del metodo, scelt tramite Armijo condition e backtracking
       - pk: np.array = Direzione di movimento della k esima iterazione del metodo
       - Ek : np.ndarray = Matrice di correzione della k esima iterazione
       - Bk : np.ndarray = Matrice postive definite generata alla k esima iterazione dall'hessiana a cui viene sommata Ek
       Metodi:
       - Step: tuple(np.array,float,np.array) --> np.array = effettua il passo di iterazione del metodo, ritorna xk+1 = xk dell'iterazione successiva
       - Backtracking: np.array --> float = effettua un passo dell'iterazione del backtracking, ritorna alpha k
       - ArmijoConditions_notmet: tuple(np.array,float) --> Bool = ritorna vero se la funzione valutata in xk + 1 è maggiore delle armijo conditions in xk
       - StoppingCriterion_notmet: np.array --> Bool= ritorna vero se k<kmax e se gradf(xk)> tolgrad 
       - CG_Find_pk: (np.array, np.ndarray) --> np.array = Risolve il sistema lineare per trovare pk che minimizza il modello quadratico, con PCG
       - Build_Bk: np.array --> np.ndarray = Costruisce una matrice Bk modificando l'essiana con una matrice Ek definita opportunamente.
       - H_is_positive_definite: np.array --> np.array or None= Effettua la cholesky factoritation per controllare se la matrice è positive definite,
            ritorna: la incomploite factorization della hessiana se si vuole risolvere il sistema lineare con il PCG
                     la complete factoritation della hessiana se si vuole risolvere il sistema lineare con la matrice fattorizzata;
                     la matrice bk, cioè la matrice hessiana modificata, qualora la matrice hessiana è non positive definite
       - Run: tuple(xk,pk,alphak) --> tuple(xk,fxk,norm_gradfxk,k,x_seq,bt_seq)= lancia il metodo
       """
    def __init__(self, x0: np.array,alpha0: float, function: str, kmax: int, tolgrad: float, c1: float, 
                 rho: float, btmax: int, solver_linear_system: str,H_correction_factor,precond: str,
                 derivatives: str, derivative_method: str, perturbation: float):
        
        self.function = function
        functions = FunctionDefinition()
        match self.function:
            case 'extended_rosenbrock':
                self.objective_function = functions.extended_rosenbrock            
            case 'discrete_boundary_value_problem':
                self.objective_function = functions.dbv_function                      
            case 'broyden_tridiagonal_function':
                self.objective_function = functions.btf_function
            case 'rosenbrock':
                self.objective_function = functions.rosenbrock_function                      

        self.x0= x0
        self.alpha0= alpha0
        self.kmax= kmax
        self.tolgrad= tolgrad
        self.c1= c1
        self.rho= rho
        self.btmax= btmax
        self.solver_linear_system= solver_linear_system
        self.H_correction_factor= H_correction_factor
        self.precond = precond
        self.k= 0
        self.bt= 0
        self.x_seq= [x0]
        self.bt_seq= []
        
        self.derivatives = derivatives
        self.d_method = derivative_method
        self.perturbation = perturbation
        self.exact_d = ExactDerivatives()
        self.finit_d = ApproximateDerivatives(self.objective_function,self.d_method, self.perturbation)
        self.sp_finit_d = SparseApproximativeDerivatives(self.objective_function, self.d_method)
        self.compute_gradient, self.compute_hessian = self.compute_gradient_hessian(self.x0)

    def compute_gradient_hessian(self, xk: np.array):
        """Define if the gradient,hessian are either exact or computed with finite difference"""
        if self.derivatives == 'exact':
            match self.function:
                case 'extended_rosenbrock':
                    return self.exact_d.extended_rosenbrock
                case 'discrete_boundary_value_problem':
                    return self.exact_d.discrete_boundary_value_problem
                case 'broyden_tridiagonal_function':
                    return self.exact_d.Broyden_tridiagonal_function
                case 'rosenbrock':
                    return self.exact_d.exact_rosenbrock
        elif self.derivatives == 'finite_differences':
            grad = self.sp_finit_d.approximate_gradient_parallel
            if len(xk) < 10**3:
                hessian = self.finit_d.hessian
                return grad,hessian
            else:
                if self.function == 'extended_rosenbrock':
                    hessian = self.sp_finit_d.hessian_approx_extendedros
                    return grad,hessian
                else: 
                    hessian = self.sp_finit_d.hessian_approx_tridiagonal
                    return grad,hessian
            

    def Step(self,xk: np.array, alphak: float, pk: np.array) -> np.array:
        xk_1= xk+ alphak*pk
        self.x_seq.append(xk_1)
        return xk_1

    def CG_Find_pk(self,xk: np.array,bk:np.array,gradf: np.array) -> np.array:
        if self.precond == 'yes':
            L = self.ichol(bk)

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
    
    def ichol(self,A,drop_tol : float = 0.0):
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

    def StoppingCriterion_notmet(self,xk: np.array, gradf)-> bool:
        return self.k<self.kmax and np.linalg.norm(gradf)> self.tolgrad
    
    def ArmijoConditions_notmet(self,xk: np.array,alphak: float, pk: np.array, gradf: np.array)-> bool:
        xk_1= xk + alphak * pk
        return self.objective_function(xk_1) > self.objective_function(xk) + self.c1 * alphak * (gradf.T @ pk)
    
    def Backtracking(self, xk: np.array, pk: np.array,gradf: np.array)-> float:
        alphak= self.alpha0
        while self.ArmijoConditions_notmet(xk,alphak,pk,gradf) and self.bt<self.btmax:
            alphak_1= self.rho*alphak
            alphak= alphak_1
            self.bt += 1
            if self.bt > self.btmax:
                return None
        self.bt_seq.append(alphak)
        return alphak
    
    def H_is_positive_definite(self,hessf) -> np.array:
        try:
            return sci.sparse.linalg.cholesky(hessf), hessf
        except np.linalg.LinAlgError:
            L, bk= self.Build_bk(hessf)
            return L, bk
                  
    def Build_bk(self,hessf: np.array) -> np.ndarray:
        beta= 10e-3
        diag_matrix= [hessf[i][i] for i in range(hessf.shape[0])]
        if min(diag_matrix) > 0:
            tau0= 0
        else:
            tau0= -min(diag_matrix) + beta 
        tauk= tau0
        bk= hessf + tauk* np.identity(hessf.shape[0])
        flag = False
        i = 0
        while flag == False and i < self.kmax:
            tauk_1= max(self.H_correction_factor*tauk,beta)
            bk_1= bk + tauk_1*np.identity(bk.shape[0])
            bk= bk_1
            tauk= tauk_1
            try:
                L= np.linalg.cholesky(bk)
                flag= True

                return L, bk
            
            except np.linalg.LinAlgError:
                pass
                flag= False
            i += 1
        if flag == False:
            print(f"Hessian can't be modified with {self.kmax}")
            exit
    
    def make_symmetric(self,Hessf: np.array,xk: np.array):
        hessf= Hessf.copy()
        if np.allclose(hessf,hessf.T, atol= 10e-8) == False:
            return 0.5 * (hessf + hessf.T)
        else:
            return hessf
                
    def Run(self)-> tuple[np.array, float, float, int, list[np.array], list[float]]:
        xk= self.x0
        grad = self.compute_gradient(xk)
        i = 0
        while self.StoppingCriterion_notmet(xk, grad):
            if i != 0:                                                      #Avoid computing the same gradient two times
                grad = self.compute_gradient(xk)
            hessf = self.compute_hessian(xk)
            if self.derivatives == 'finite_differences':
                hessf = self.make_symmetric(hessf,xk)                       #Approximation could make hessian not symmetric
            L, bk= self.H_is_positive_definite(hessf)                       #bk==hessf if hessian is positive definite
            if self.solver_linear_system == 'cg':
                pk= self.CG_Find_pk(xk,bk,grad)
            elif self.solver_linear_system == 'chol':
                if len(xk) < 10**3:
                    pk= self.chol_Find_Pk(L,grad)
                else:
                    print(f"Is not possible to find pk with cholesky with dimension {len(xk)}")
                    exit
            alphak= self.Backtracking(xk,pk,grad)
            if alphak == None:
                print(f"La backtracking strategy non è riuscita a trovare aplhak che soddifsi le Armijo Conditions con {self.btmax} iterazioni")
                print(f"Il metodo non converge")
                exit
            else:    
                xk_1= self.Step(xk,alphak,pk)
                self.x_seq.append(xk_1)
                xk= xk_1
                self.k += 1
            i += 1
        
        norm_gradfxk= np.linalg.norm(grad)

        return xk,self.objective_function(xk),norm_gradfxk, self.k, self.x_seq, self.bt_seq