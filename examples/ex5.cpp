//                                MFEM Example 5
//
//
// Description:  This example code demonstrates the use of MFEM to define a 
//               triangulation of a unit sphere and a simple isoparametric
//               finite element discretization of the Laplace problem 
//               -Delta u + u = f on a triangulated sphere using linear finite
//               elements.  The example highlights mesh generation, the use of
//               mesh refinement, finite element grid functions, as well as
//               linear and bilinear forms corresponding to the left-hand side
//               and right-hand side of the discrete linear system.

#include <fstream>
#include "mfem.hpp"
#include <cassert>

double analytic_solution (Vector & input)
{
  double l2 = (input[0]*input[0] + input[1]*input[1] +input[2]*input[2]);
  return input[0]*input[1]/l2;
}


double analytic_rhs (Vector & input)
{
  double l2 = (input[0]*input[0] + input[1]*input[1] +input[2]*input[2]);
  return 7*input[0]*input[1]/l2;
}

int main (int argc, char *argv[])
{
   if (argc == 1)
   {
      cout << "\nUsage: ex5 <number_of_refinements>\n" << endl;
      return 1;
   }

  // Mesh(int _Dim, int NVert, int NElem, int NBdrElem = 0, int _spaceDim= -1)
  const int Nvert = 6;
  const int NElem = 8;

  Mesh mesh(2, Nvert, NElem, 0, 3);

   // Sets vertices and the corresponding coordinates
   double v[Nvert*3] =
   {
     1,             0,             0,
     0,             1,             0,
     -1,            0,             0,
     0,             -1,            0,
     0,             0,             1,
     0,             0,            -1
   };


  for (int j = 0; j < Nvert; j++)
  {
    mesh.AddVertex(v+3*j);
  }

  int e[NElem*3] = 
  {
    0, 1, 4,
    1, 2, 4,
    2, 3, 4,
    3, 0, 4,
    1, 0, 5,
    2, 1, 5,
    3, 2, 5,
    0, 3, 5
  };

  // Sets the elements and the corresponding indices of vertices
  for (int j = 0; j < NElem; j++)
  {
    mesh.AddTriangle(e+3*j,1);
  }

  mesh.FinalizeTriMesh(1,1,1);

  // Refine the mesh 
  const int ref_levels = atoi(argv[1]);
  for (int l = 0; l < ref_levels; l++)
  {
    mesh.UniformRefinement();

    // snap the new vertices of the refined mesh back to sphere surface
    for (int i=0; i<mesh.GetNV(); i++)
    {
      double * coord = mesh.GetVertex(i);
      double l = sqrt(coord[0]*coord[0] + coord[1]*coord[1] + coord[2]*coord[2]);
      assert (l>0);
      for (int j=0; j<3; j++)
	coord[j] /= l;
    }
  }

   // 3. Define a finite element space on the mesh. Here we use isoparametric
   //    finite elements coming from the mesh nodes (linear by default).
   FiniteElementCollection *fec;
   if (mesh.GetNodes())
      fec = mesh.GetNodes()->OwnFEC();
   else
      fec = new LinearFECollection;
   FiniteElementSpace *fespace = new FiniteElementSpace(&mesh, fec);
   cout << "Number of unknowns: " << fespace->GetVSize() << endl;

   // 4. Set up the linear form b(.) which corresponds to the right-hand side of
   //    the FEM linear system, which in this case is (1,phi_i) where phi_i are
   //    the basis functions in the finite element fespace.
   LinearForm *b = new LinearForm(fespace);
   ConstantCoefficient one(1.0);
   FunctionCoefficient rhs_coef (analytic_rhs);
   FunctionCoefficient sol_coef (analytic_solution);

   b->AddDomainIntegrator(new DomainLFIntegrator(rhs_coef));
   b->Assemble();

   // 5. Define the solution vector x as a finite element grid function
   //    corresponding to fespace. Initialize x with initial guess of zero,
   //    which satisfies the boundary conditions.
   GridFunction x(fespace);
   x = 0.0;

   // 6. Set up the bilinear form a(.,.) on the finite element space
   //    corresponding to the Laplacian operator -Delta, by adding the Diffusion
   //    domain integrator and imposing homogeneous Dirichlet boundary
   //    conditions. The boundary conditions are implemented by marking all the
   //    boundary attributes from the mesh as essential (Dirichlet). After
   //    assembly and finalizing we extract the corresponding sparse matrix A.
   BilinearForm *a = new BilinearForm(fespace);
   a->AddDomainIntegrator(new DiffusionIntegrator(one));
   a->AddDomainIntegrator(new MassIntegrator(one));
   a->Assemble();
   a->Finalize();
   const SparseMatrix &A = a->SpMat();

#ifndef MFEM_USE_SUITESPARSE
   // 7. Define a simple symmetric Gauss-Seidel preconditioner and use it to
   //    solve the system Ax=b with PCG.
   GSSmoother M(A);
   PCG(A, M, *b, x, 0, 10000, 1e-12, 0.0);
#else
   // 7. If MFEM was compiled with SuiteSparse, use UMFPACK to solve the system.
   UMFPackSolver umf_solver;
   umf_solver.Control[UMFPACK_ORDERING] = UMFPACK_ORDERING_METIS;
   umf_solver.SetOperator(A);
   umf_solver.Mult(*b, x);
#endif

   // compare with exact solution
   cout<<"L2 norm of error: " << x.ComputeL2Error(sol_coef) << endl;

   // 8. Save the solution and the mesh. 
   {
      ofstream mesh_ofs("sphere_refined.mesh");
      mesh_ofs.precision(8);
      mesh.Print(mesh_ofs);
      ofstream sol_ofs("sol.gf");
      sol_ofs.precision(8);
      x.Save(sol_ofs);
   }

   // 10. Free the used memory.
   delete a;
   delete b;
   delete fespace;
   if (!mesh.GetNodes())
      delete fec;
   return 0;
}
