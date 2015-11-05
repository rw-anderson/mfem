// Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.

#ifndef MFEM_HYPRE
#define MFEM_HYPRE

#include "../config/config.hpp"

#ifdef MFEM_USE_MPI

#include <mpi.h>

// Enable internal hypre timing routines
#define HYPRE_TIMING

// hypre header files
#include "seq_mv.h"
#include "_hypre_parcsr_mv.h"
#include "_hypre_parcsr_ls.h"
#include "temp_multivector.h"

#include "sparsemat.hpp"

namespace mfem
{

class ParFiniteElementSpace;
class HypreParMatrix;

namespace internal
{

// Convert a HYPRE_Int to int
inline int to_int(HYPRE_Int i)
{
#ifdef HYPRE_BIGINT
   MFEM_ASSERT(HYPRE_Int(int(i)) == i, "overflow converting HYPRE_Int to int");
#endif
   return int(i);
}

}

/// Wrapper for hypre's parallel vector class
class HypreParVector : public Vector
{
private:
   int own_ParVector;

   /// The actual object
   hypre_ParVector *x;

   friend class HypreParMatrix;

   // Set Vector::data and Vector::size from *x
   inline void _SetDataAndSize_();

public:
   /** Creates vector with given global size and partitioning of the columns.
       Processor P owns columns [col[P],col[P+1]) */
   HypreParVector(MPI_Comm comm, HYPRE_Int glob_size, HYPRE_Int *col);
   /** Creates vector with given global size, partitioning of the columns,
       and data. The data must be allocated and destroyed outside. */
   HypreParVector(MPI_Comm comm, HYPRE_Int glob_size, double *_data,
                  HYPRE_Int *col);
   /// Creates vector compatible with y
   HypreParVector(const HypreParVector &y);
   /// Creates vector compatible with (i.e. in the domain of) A or A^T
   HypreParVector(HypreParMatrix &A, int tr = 0);
   /// Creates vector wrapping y
   HypreParVector(HYPRE_ParVector y);
   /// Create a true dof parallel vector on a given ParFiniteElementSpace
   HypreParVector(ParFiniteElementSpace *pfes);

   /// MPI communicator
   MPI_Comm GetComm() { return x->comm; }

   /// Returns the row partitioning
   inline HYPRE_Int *Partitioning() { return x->partitioning; }

   /// Returns the global number of rows
   inline HYPRE_Int GlobalSize() { return x->global_size; }

   /// Typecasting to hypre's hypre_ParVector*
   operator hypre_ParVector*() const;
#ifndef HYPRE_PAR_VECTOR_STRUCT
   /// Typecasting to hypre's HYPRE_ParVector, a.k.a. void *
   operator HYPRE_ParVector() const;
#endif
   /// Changes the ownership of the the vector
   hypre_ParVector *StealParVector() { own_ParVector = 0; return x; }

   /// Returns the global vector in each processor
   Vector *GlobalVector();

   /// Set constant values
   HypreParVector& operator= (double d);
   /// Define '=' for hypre vectors.
   HypreParVector& operator= (const HypreParVector &y);

   /** Sets the data of the Vector and the hypre_ParVector to _data.
       Must be used only for HypreParVectors that do not own the data,
       e.g. created with the constructor:
       HypreParVector(int glob_size, double *_data, int *col).  */
   void SetData(double *_data);

   /// Set random values
   HYPRE_Int Randomize(HYPRE_Int seed);

   /// Prints the locally owned rows in parallel
   void Print(const char *fname);

   /// Calls hypre's destroy function
   ~HypreParVector();
};

/// Returns the inner product of x and y
double InnerProduct(HypreParVector &x, HypreParVector &y);
double InnerProduct(HypreParVector *x, HypreParVector *y);


/// Wrapper for hypre's ParCSR matrix class
class HypreParMatrix : public Operator
{
private:
   /// The actual object
   hypre_ParCSRMatrix *A;

   /// Auxiliary vectors for typecasting
   mutable HypreParVector *X, *Y;

   // Flags indicating ownership of A->diag->{i,j,data}, A->offd->{i,j,data},
   // and A->col_map_offd.
   // The possible values for diagOwner are:
   //  -1: no special treatment of A->diag (default)
   //   0: prevent hypre from destroying A->diag->{i,j,data}
   //   1: same as 0, plus take ownership of A->diag->{i,j}
   //   2: same as 0, plus take ownership of A->diag->data
   //   3: same as 0, plus take ownership of A->diag->{i,j,data}
   // The same values and rules apply to offdOwner and A->offd.
   // The possible values for colMapOwner are:
   //  -1: no special treatment of A->col_map_offd (default)
   //   0: prevent hypre from destroying A->col_map_offd
   //   1: same as 0, plus take ownership of A->col_map_offd
   // All owned arrays are destroyed with 'delete []'.
   char diagOwner, offdOwner, colMapOwner;

   // Initialize with defaults. Does not initalize inherited members.
   void Init();

   // Delete all owned data. Does not perform re-initialization with defaults.
   void Destroy();

   // Copy (shallow or deep, based on HYPRE_BIGINT) the I and J arrays from csr
   // to hypre_csr. Shallow copy the data. Return the appropriate ownership
   // flag.
   static char CopyCSR(SparseMatrix *csr, hypre_CSRMatrix *hypre_csr);
   // Copy (shallow or deep, based on HYPRE_BIGINT) the I and J arrays from
   // bool_csr to hypre_csr. Allocate the data array and set it to all ones.
   // Return the appropriate ownership flag.
   static char CopyBoolCSR(Table *bool_csr, hypre_CSRMatrix *hypre_csr);

   // Copy the j array of a hypre_CSRMatrix to the given J array, converting
   // the indices from HYPRE_Int to int.
   static void CopyCSR_J(hypre_CSRMatrix *hypre_csr, int *J);

public:
   /// Converts hypre's format to HypreParMatrix
   HypreParMatrix(hypre_ParCSRMatrix *a)
   { Init(); A = a; height = GetNumRows(); width = GetNumCols(); }
   /** Creates block-diagonal square parallel matrix. Diagonal is given by diag
       which must be in CSR format (finalized). The new HypreParMatrix does not
       take ownership of any of the input arrays. */
   HypreParMatrix(MPI_Comm comm, HYPRE_Int glob_size, HYPRE_Int *row_starts,
                  SparseMatrix *diag);
   /** Creates block-diagonal rectangular parallel matrix. Diagonal is given by
       diag which must be in CSR format (finalized). The new HypreParMatrix does
       not take ownership of any of the input arrays. */
   HypreParMatrix(MPI_Comm comm, HYPRE_Int global_num_rows,
                  HYPRE_Int global_num_cols, HYPRE_Int *row_starts,
                  HYPRE_Int *col_starts, SparseMatrix *diag);
   /** Creates general (rectangular) parallel matrix. The new HypreParMatrix
       does not take ownership of any of the input arrays. */
   HypreParMatrix(MPI_Comm comm, HYPRE_Int global_num_rows,
                  HYPRE_Int global_num_cols, HYPRE_Int *row_starts,
                  HYPRE_Int *col_starts, SparseMatrix *diag, SparseMatrix *offd,
                  HYPRE_Int *cmap);
   /** Creates general (rectangular) parallel matrix. The new HypreParMatrix
       takes ownership of all input arrays, except col_starts and row_starts. */
   HypreParMatrix(MPI_Comm comm,
                  HYPRE_Int global_num_rows, HYPRE_Int global_num_cols,
                  HYPRE_Int *row_starts, HYPRE_Int *col_starts,
                  HYPRE_Int *diag_i, HYPRE_Int *diag_j, double *diag_data,
                  HYPRE_Int *offd_i, HYPRE_Int *offd_j, double *offd_data,
                  HYPRE_Int offd_num_cols, HYPRE_Int *offd_col_map);

   /// Creates a parallel matrix from SparseMatrix on processor 0.
   HypreParMatrix(MPI_Comm comm, HYPRE_Int *row_starts, HYPRE_Int *col_starts,
                  SparseMatrix *a);

   /** Creates boolean block-diagonal rectangular parallel matrix. The new
       HypreParMatrix does not take ownership of any of the input arrays. */
   HypreParMatrix(MPI_Comm comm, HYPRE_Int global_num_rows,
                  HYPRE_Int global_num_cols, HYPRE_Int *row_starts,
                  HYPRE_Int *col_starts, Table *diag);
   /** Creates boolean rectangular parallel matrix. The new HypreParMatrix takes
       ownership of the arrays i_diag, j_diag, i_offd, j_offd, and cmap; does
       not take ownership of the arrays row and col. */
   HypreParMatrix(MPI_Comm comm, int id, int np, HYPRE_Int *row, HYPRE_Int *col,
                  HYPRE_Int *i_diag, HYPRE_Int *j_diag, HYPRE_Int *i_offd,
                  HYPRE_Int *j_offd, HYPRE_Int *cmap, HYPRE_Int cmap_size);

   /** Creates a general parallel matrix from a local CSR matrix on each
       processor described by the I, J and data arrays. The local matrix should
       be of size (local) nrows by (global) glob_ncols. The new parallel matrix
       contains copies of all input arrays (so they can be deleted). */
   HypreParMatrix(MPI_Comm comm, int nrows, HYPRE_Int glob_nrows,
                  HYPRE_Int glob_ncols, int *I, HYPRE_Int *J,
                  double *data, HYPRE_Int *rows, HYPRE_Int *cols);

   /// MPI communicator
   MPI_Comm GetComm() { return A->comm; }

   /// Typecasting to hypre's hypre_ParCSRMatrix*
   operator hypre_ParCSRMatrix*() { return A; }
#ifndef HYPRE_PAR_CSR_MATRIX_STRUCT
   /// Typecasting to hypre's HYPRE_ParCSRMatrix, a.k.a. void *
   operator HYPRE_ParCSRMatrix() { return (HYPRE_ParCSRMatrix) A; }
#endif
   /// Changes the ownership of the the matrix
   hypre_ParCSRMatrix* StealData();

   /** If the HypreParMatrix does not own the row-starts array, make a copy of
       it that the HypreParMatrix will own. If the col-starts array is the same
       as the row-starts array, col-starts is also replaced. */
   void CopyRowStarts();
   /** If the HypreParMatrix does not own the col-starts array, make a copy of
       it that the HypreParMatrix will own. If the row-starts array is the same
       as the col-starts array, row-starts is also replaced. */
   void CopyColStarts();

   /// Returns the global number of nonzeros
   inline HYPRE_Int NNZ() { return A->num_nonzeros; }
   /// Returns the row partitioning
   inline HYPRE_Int *RowPart() { return A->row_starts; }
   /// Returns the column partitioning
   inline HYPRE_Int *ColPart() { return A->col_starts; }
   /// Returns the global number of rows
   inline HYPRE_Int M() { return A->global_num_rows; }
   /// Returns the global number of columns
   inline HYPRE_Int N() { return A->global_num_cols; }

   /// Get the diagonal of the matrix
   void GetDiag(Vector &diag);
   /// Returns the transpose of *this
   HypreParMatrix * Transpose();

   /// Returns the number of rows in the diagonal block of the ParCSRMatrix
   int GetNumRows() const
   {
      return internal::to_int(
                hypre_CSRMatrixNumRows(hypre_ParCSRMatrixDiag(A)));
   }

   /// Returns the number of columns in the diagonal block of the ParCSRMatrix
   int GetNumCols() const
   {
      return internal::to_int(
                hypre_CSRMatrixNumCols(hypre_ParCSRMatrixDiag(A)));
   }

   HYPRE_Int GetGlobalNumRows() const
   { return hypre_ParCSRMatrixGlobalNumRows(A); }

   HYPRE_Int GetGlobalNumCols() const
   { return hypre_ParCSRMatrixGlobalNumCols(A); }

   HYPRE_Int *GetRowStarts() const { return hypre_ParCSRMatrixRowStarts(A); }

   HYPRE_Int *GetColStarts() const { return hypre_ParCSRMatrixColStarts(A); }

   /// Computes y = alpha * A * x + beta * y
   HYPRE_Int Mult(HypreParVector &x, HypreParVector &y,
                  double alpha = 1.0, double beta = 0.0);
   /// Computes y = alpha * A * x + beta * y
   HYPRE_Int Mult(HYPRE_ParVector x, HYPRE_ParVector y,
                  double alpha = 1.0, double beta = 0.0);
   /// Computes y = alpha * A^t * x + beta * y
   HYPRE_Int MultTranspose(HypreParVector &x, HypreParVector &y,
                           double alpha = 1.0, double beta = 0.0);

   void Mult(double a, const Vector &x, double b, Vector &y) const;
   void MultTranspose(double a, const Vector &x, double b, Vector &y) const;

   virtual void Mult(const Vector &x, Vector &y) const
   { Mult(1.0, x, 0.0, y); }
   virtual void MultTranspose(const Vector &x, Vector &y) const
   { MultTranspose(1.0, x, 0.0, y); }

   /// Scale the local row i by s(i).
   void ScaleRows(const Vector & s);
   /// Scale the local row i by 1./s(i)
   void InvScaleRows(const Vector & s);
   /// Scale all entries by s: A_scaled = s*A.
   void operator*=(double s);

   /// If a row contains only zeros, set its diagonal to 1.
   void EliminateZeroRows() { hypre_ParCSRMatrixFixZeroRows(A); }

   /// Prints the locally owned rows in parallel
   void Print(const char *fname, HYPRE_Int offi = 0, HYPRE_Int offj = 0);
   /// Reads the matrix from a file
   void Read(MPI_Comm comm, const char *fname);

   /// Calls hypre's destroy function
   virtual ~HypreParMatrix() { Destroy(); }
};

/// Returns the matrix A * B
HypreParMatrix * ParMult(HypreParMatrix *A, HypreParMatrix *B);

/// Returns the matrix P^t * A * P
HypreParMatrix * RAP(HypreParMatrix *A, HypreParMatrix *P);
/// Returns the matrix Rt^t * A * P
HypreParMatrix * RAP(HypreParMatrix * Rt, HypreParMatrix *A, HypreParMatrix *P);

/** Eliminate essential b.c. specified by ess_dof_list from the solution x to
    the r.h.s. b. Here A is matrix with eliminated b.c., while Ae is such that
    (A+Ae) is the original (Neumann) matrix before elimination. */
void EliminateBC(HypreParMatrix &A, HypreParMatrix &Ae,
                 Array<int> &ess_dof_list,
                 HypreParVector &x, HypreParVector &b);


/// Parallel smoothers in hypre
class HypreSmoother : public Solver
{
protected:
   /// The linear system matrix
   HypreParMatrix *A;
   /// Right-hand side and solution vectors
   mutable HypreParVector *B, *X;
   /// Temporary vectors
   mutable HypreParVector *V, *Z;
   /// FIR Filter Temporary Vectors
   mutable HypreParVector *X0, *X1;

   /** Smoother type from hypre_ParCSRRelax() in ams.c plus extensions, see the
       enumeartion Type below. */
   int type;
   /// Number of relaxation sweeps
   int relax_times;
   /// Damping coefficient (usually <= 1)
   double relax_weight;
   /// SOR parameter (usually in (0,2))
   double omega;
   /// Order of the smoothing polynomial
   int poly_order;
   /// Fraction of spectrum to smooth for polynomial relaxation
   double poly_fraction;
   /// Apply the polynomial smoother to A or D^{-1/2} A D^{-1/2}
   int poly_scale;

   /// Taubin's lambda-mu method parameters
   double lambda;
   double mu;
   int taubin_iter;

   /// l1 norms of the rows of A
   double *l1_norms;
   /// Maximal eigenvalue estimate for polynomial smoothing
   double max_eig_est;
   /// Minimal eigenvalue estimate for polynomial smoothing
   double min_eig_est;
   /// Paramters for windowing function of FIR filter
   double window_params[3];

   /// Combined coefficients for windowing and Chebyshev polynomials.
   double* fir_coeffs;

public:
   /** Hypre smoother types:
       0    = Jacobi
       1    = l1-scaled Jacobi
       2    = l1-scaled block Gauss-Seidel/SSOR
       4    = truncated l1-scaled block Gauss-Seidel/SSOR
       5    = lumped Jacobi
       6    = Gauss-Seidel
       16   = Chebyshev
       1001 = Taubin polynomial smoother
       1002 = FIR polynomial smoother. */
   enum Type { Jacobi = 0, l1Jacobi = 1, l1GS = 2, l1GStr = 4, lumpedJacobi = 5,
               GS = 6, Chebyshev = 16, Taubin = 1001, FIR = 1002
             };

   HypreSmoother();

   HypreSmoother(HypreParMatrix &_A, int type = l1GS,
                 int relax_times = 1, double relax_weight = 1.0,
                 double omega = 1.0, int poly_order = 2,
                 double poly_fraction = .3);

   /// Set the relaxation type and number of sweeps
   void SetType(HypreSmoother::Type type, int relax_times = 1);
   /// Set SOR-related parameters
   void SetSOROptions(double relax_weight, double omega);
   /// Set parameters for polynomial smoothing
   void SetPolyOptions(int poly_order, double poly_fraction);
   /// Set parameters for Taubin's lambda-mu method
   void SetTaubinOptions(double lambda, double mu, int iter);

   /// Convenience function for setting canonical windowing parameters
   void SetWindowByName(const char* window_name);
   /// Set parameters for windowing function for FIR smoother.
   void SetWindowParameters(double a, double b, double c);
   /// Compute window and Chebyshev coefficients for given polynomial order.
   void SetFIRCoefficients(double max_eig);

   /** Set/update the associated operator. Mult be called after setting the
       HypreSmoother type and options. */
   virtual void SetOperator(const Operator &op);

   /// Relax the linear system Ax=b
   virtual void Mult(const HypreParVector &b, HypreParVector &x) const;
   virtual void Mult(const Vector &b, Vector &x) const;

   virtual ~HypreSmoother();
};


/// Abstract class for hypre's solvers and preconditioners
class HypreSolver : public Solver
{
protected:
   /// The linear system matrix
   HypreParMatrix *A;

   /// Right-hand side and solution vector
   mutable HypreParVector *B, *X;

   /// Was hypre's Setup function called already?
   mutable int setup_called;

public:
   HypreSolver();

   HypreSolver(HypreParMatrix *_A);

   /// Typecast to HYPRE_Solver -- return the solver
   virtual operator HYPRE_Solver() const = 0;

   /// hypre's internal Setup function
   virtual HYPRE_PtrToParSolverFcn SetupFcn() const = 0;
   /// hypre's internal Solve function
   virtual HYPRE_PtrToParSolverFcn SolveFcn() const = 0;

   virtual void SetOperator(const Operator &op)
   { mfem_error("HypreSolvers do not support SetOperator!"); }

   /// Solve the linear system Ax=b
   virtual void Mult(const HypreParVector &b, HypreParVector &x) const;
   virtual void Mult(const Vector &b, Vector &x) const;

   virtual ~HypreSolver();
};

/// PCG solver in hypre
class HyprePCG : public HypreSolver
{
private:
   int print_level;
   HYPRE_Solver pcg_solver;

public:
   HyprePCG(HypreParMatrix &_A);

   void SetTol(double tol);
   void SetMaxIter(int max_iter);
   void SetLogging(int logging);
   void SetPrintLevel(int print_lvl);

   /// Set the hypre solver to be used as a preconditioner
   void SetPreconditioner(HypreSolver &precond);

   /** Use the L2 norm of the residual for measuring PCG convergence, plus
       (optionally) 1) periodically recompute true residuals from scratch; and
       2) enable residual-based stopping criteria. */
   void SetResidualConvergenceOptions(int res_frequency=-1, double rtol=0.0);

   /// non-hypre setting
   void SetZeroInintialIterate() { iterative_mode = false; }

   void GetNumIterations(int &num_iterations)
   {
      HYPRE_Int num_it;
      HYPRE_ParCSRPCGGetNumIterations(pcg_solver, &num_it);
      num_iterations = internal::to_int(num_it);
   }

   /// The typecast to HYPRE_Solver returns the internal pcg_solver
   virtual operator HYPRE_Solver() const { return pcg_solver; }

   /// PCG Setup function
   virtual HYPRE_PtrToParSolverFcn SetupFcn() const
   { return (HYPRE_PtrToParSolverFcn) HYPRE_ParCSRPCGSetup; }
   /// PCG Solve function
   virtual HYPRE_PtrToParSolverFcn SolveFcn() const
   { return (HYPRE_PtrToParSolverFcn) HYPRE_ParCSRPCGSolve; }

   /// Solve Ax=b with hypre's PCG
   virtual void Mult(const HypreParVector &b, HypreParVector &x) const;
   using HypreSolver::Mult;

   virtual ~HyprePCG();
};

/// GMRES solver in hypre
class HypreGMRES : public HypreSolver
{
private:
   int print_level;
   HYPRE_Solver gmres_solver;

public:
   HypreGMRES(HypreParMatrix &_A);

   void SetTol(double tol);
   void SetMaxIter(int max_iter);
   void SetKDim(int dim);
   void SetLogging(int logging);
   void SetPrintLevel(int print_lvl);

   /// Set the hypre solver to be used as a preconditioner
   void SetPreconditioner(HypreSolver &precond);

   /// non-hypre setting
   void SetZeroInintialIterate() { iterative_mode = false; }

   /// The typecast to HYPRE_Solver returns the internal gmres_solver
   virtual operator HYPRE_Solver() const  { return gmres_solver; }

   /// GMRES Setup function
   virtual HYPRE_PtrToParSolverFcn SetupFcn() const
   { return (HYPRE_PtrToParSolverFcn) HYPRE_ParCSRGMRESSetup; }
   /// GMRES Solve function
   virtual HYPRE_PtrToParSolverFcn SolveFcn() const
   { return (HYPRE_PtrToParSolverFcn) HYPRE_ParCSRGMRESSolve; }

   /// Solve Ax=b with hypre's GMRES
   virtual void Mult (const HypreParVector &b, HypreParVector &x) const;
   using HypreSolver::Mult;

   virtual ~HypreGMRES();
};

/// The identity operator as a hypre solver
class HypreIdentity : public HypreSolver
{
public:
   virtual operator HYPRE_Solver() const { return NULL; }

   virtual HYPRE_PtrToParSolverFcn SetupFcn() const
   { return (HYPRE_PtrToParSolverFcn) hypre_ParKrylovIdentitySetup; }
   virtual HYPRE_PtrToParSolverFcn SolveFcn() const
   { return (HYPRE_PtrToParSolverFcn) hypre_ParKrylovIdentity; }

   virtual ~HypreIdentity() { }
};

/// Jacobi preconditioner in hypre
class HypreDiagScale : public HypreSolver
{
public:
   HypreDiagScale() : HypreSolver() { }
   explicit HypreDiagScale(HypreParMatrix &A) : HypreSolver(&A) { }
   virtual operator HYPRE_Solver() const { return NULL; }

   virtual HYPRE_PtrToParSolverFcn SetupFcn() const
   { return (HYPRE_PtrToParSolverFcn) HYPRE_ParCSRDiagScaleSetup; }
   virtual HYPRE_PtrToParSolverFcn SolveFcn() const
   { return (HYPRE_PtrToParSolverFcn) HYPRE_ParCSRDiagScale; }

   HypreParMatrix* GetData() { return A; }
   virtual ~HypreDiagScale() { }
};

/// The ParaSails preconditioner in hypre
class HypreParaSails : public HypreSolver
{
private:
   HYPRE_Solver sai_precond;

public:
   HypreParaSails(HypreParMatrix &A);

   void SetSymmetry(int sym);

   /// The typecast to HYPRE_Solver returns the internal sai_precond
   virtual operator HYPRE_Solver() const { return sai_precond; }

   virtual HYPRE_PtrToParSolverFcn SetupFcn() const
   { return (HYPRE_PtrToParSolverFcn) HYPRE_ParaSailsSetup; }
   virtual HYPRE_PtrToParSolverFcn SolveFcn() const
   { return (HYPRE_PtrToParSolverFcn) HYPRE_ParaSailsSolve; }

   virtual ~HypreParaSails();
};

/// The BoomerAMG solver in hypre
class HypreBoomerAMG : public HypreSolver
{
private:
   HYPRE_Solver amg_precond;

   // If amg_precond is NULL, allocates it and sets default options.
   // Otherwise saves the options from amg_precond, destroys it, allocates a new
   // one, and sets its options to the saved values.
   void ResetAMGPrecond();

public:
   HypreBoomerAMG();

   HypreBoomerAMG(HypreParMatrix &A);

   virtual void SetOperator(const Operator &op);

   /** More robust options for systems, such as elastisity. Note that BoomerAMG
       assumes Ordering::byVDIM in the finite element space used to generate the
       matrix A. */
   void SetSystemsOptions(int dim);

   void SetPrintLevel(int print_level)
   { HYPRE_BoomerAMGSetPrintLevel(amg_precond, print_level); }

   /// The typecast to HYPRE_Solver returns the internal amg_precond
   virtual operator HYPRE_Solver() const { return amg_precond; }

   virtual HYPRE_PtrToParSolverFcn SetupFcn() const
   { return (HYPRE_PtrToParSolverFcn) HYPRE_BoomerAMGSetup; }
   virtual HYPRE_PtrToParSolverFcn SolveFcn() const
   { return (HYPRE_PtrToParSolverFcn) HYPRE_BoomerAMGSolve; }

   virtual ~HypreBoomerAMG();
};

/// Compute the discrete gradient matrix between the nodal linear and ND1 spaces
HypreParMatrix* DiscreteGrad(ParFiniteElementSpace *edge_fespace,
                             ParFiniteElementSpace *vert_fespace);
/// Compute the discrete curl matrix between the ND1 and RT0 spaces
HypreParMatrix* DiscreteCurl(ParFiniteElementSpace *face_fespace,
                             ParFiniteElementSpace *edge_fespace);

/// The Auxiliary-space Maxwell Solver in hypre
class HypreAMS : public HypreSolver
{
private:
   HYPRE_Solver ams;

   /// Vertex coordinates
   HypreParVector *x, *y, *z;
   /// Discrete gradient matrix
   HypreParMatrix *G;
   /// Nedelec interpolation matrix and its components
   HypreParMatrix *Pi, *Pix, *Piy, *Piz;

public:
   HypreAMS(HypreParMatrix &A, ParFiniteElementSpace *edge_fespace,
            int singular_problem = 0);

   /// The typecast to HYPRE_Solver returns the internal ams object
   virtual operator HYPRE_Solver() const { return ams; }

   virtual HYPRE_PtrToParSolverFcn SetupFcn() const
   { return (HYPRE_PtrToParSolverFcn) HYPRE_AMSSetup; }
   virtual HYPRE_PtrToParSolverFcn SolveFcn() const
   { return (HYPRE_PtrToParSolverFcn) HYPRE_AMSSolve; }

   virtual ~HypreAMS();
};

/// The Auxiliary-space Divergence Solver in hypre
class HypreADS : public HypreSolver
{
private:
   HYPRE_Solver ads;

   /// Vertex coordinates
   HypreParVector *x, *y, *z;
   /// Discrete gradient matrix
   HypreParMatrix *G;
   /// Discrete curl matrix
   HypreParMatrix *C;
   /// Nedelec interpolation matrix and its components
   HypreParMatrix *ND_Pi, *ND_Pix, *ND_Piy, *ND_Piz;
   /// Raviart-Thomas interpolation matrix and its components
   HypreParMatrix *RT_Pi, *RT_Pix, *RT_Piy, *RT_Piz;

public:
   HypreADS(HypreParMatrix &A, ParFiniteElementSpace *face_fespace);

   /// The typecast to HYPRE_Solver returns the internal ads object
   virtual operator HYPRE_Solver() const { return ads; }

   virtual HYPRE_PtrToParSolverFcn SetupFcn() const
   { return (HYPRE_PtrToParSolverFcn) HYPRE_ADSSetup; }
   virtual HYPRE_PtrToParSolverFcn SolveFcn() const
   { return (HYPRE_PtrToParSolverFcn) HYPRE_ADSSolve; }

   virtual ~HypreADS();
};

class HypreMultiVector
{
private:
   // Pointer to hypre's multi-vector object
   mv_MultiVectorPtr mv_ptr;

   // Interface for matrix storage type
   mv_InterfaceInterpreter interpreter;

   // Wrappers for each member of the multivector
   HypreParVector ** hpv;

   // Number of vectors in the multivector
   int nv;

public:
   HypreMultiVector(int n, HypreParVector & v);
   ~HypreMultiVector();

   /// Set random values
   void Randomize(HYPRE_Int seed);

   /// Extract a single HypreParVector object
   HypreParVector & GetVector(unsigned int i);

   operator mv_MultiVectorPtr() const { return mv_ptr; }

   mv_InterfaceInterpreter & GetInterpreter() { return interpreter; }
   mv_MultiVectorPtr       & GetMultiVector() { return mv_ptr; }
};

/// LOBPCG eigenvalue solver in hypre
class HypreLOBPCG
{
private:
   // Pointer to HYPRE's solver struct
   HYPRE_Solver lobpcg_solver;

   // Interface for setting up and performing matrix-vector products
   HYPRE_MatvecFunctions matvec_fn;

public:
   HypreLOBPCG(mv_InterfaceInterpreter & interpreter);
   ~HypreLOBPCG();

   void SetTol(double tol);
   void SetMaxIter(int max_iter);
   void SetPrintLevel(int logging);
   void SetPrecondUsageMode(int pcg_mode);

   /// The following four methods support linear systems made up of
   /// simple HypreParMatrices
   void SetPrecond(HypreSolver & precond);
   void Setup(HypreParMatrix & A, HypreParVector & b, HypreParVector & x);
   void SetupB(HypreParMatrix & B, HypreParVector & x);
   void SetupT(HypreParMatrix & T, HypreParVector & x);

   /// The following four methods support more general linear systems
   void SetPrecond(Solver & precond);
   void Setup(Operator & A, HypreParVector & b, HypreParVector & x);
   void SetupB(Operator & B, HypreParVector & x);
   void SetupT(Operator & T, HypreParVector & x);

   void Solve(Vector & eigenvalues);
   void Solve(Vector & eigenvalues, HypreMultiVector & eigenvectors);
   void Solve(Vector & eigenvalues, HypreMultiVector & eigenvectors,
              HypreMultiVector & constraints);

   static void    * BlockOperatorMatvecCreate( void *A, void *x );
   static HYPRE_Int BlockOperatorMatvec( void *matvec_data,
                                         HYPRE_Complex alpha,
                                         void *A,
                                         void *x,
                                         HYPRE_Complex beta,
                                         void *y );
   static HYPRE_Int BlockOperatorMatvecDestroy( void *matvec_data );

   static HYPRE_Int BlockDiagonalPrecondSolve(void *solver,
                                              void *A,
                                              void *b,
                                              void *x);
   static HYPRE_Int BlockDiagonalPrecondSetup(void *solver,
                                              void *A,
                                              void *b,
                                              void *x);



};

}

#endif // MFEM_USE_MPI

#endif
