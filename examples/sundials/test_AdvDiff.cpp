// This is a 2D analog of the AdvDiff_ASA_p_non_p.c SUNDIALS CVODES example

#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <algorithm>

#ifndef MFEM_USE_SUNDIALS
#error This example requires that MFEM is built with MFEM_USE_SUNDIALS=YES
#endif

using namespace std;
using namespace mfem;

// Choice for the problem setup. The fluid velocity, initial condition and
// inflow boundary condition are chosen based on this parameter.
int problem;

// Mesh bounding box
Vector bb_min, bb_max;


/** Reimplement AdvDiff problem here */
class AdvDiffSUNDIALS : public TimeDependentAdjointOperator
{
public:
  AdvDiffSUNDIALS(int dim, Vector p, ParFiniteElementSpace *fes) :
    TimeDependentAdjointOperator(dim),
    p_(p),
    adjointMatrix(NULL),
    M(NULL), K(NULL),
    m(NULL), k(NULL),
    pfes(fes),
    M_solver(fes->GetComm())
  {
    int skip_zeros = 0;
    ParMesh * pmesh = pfes->GetParMesh();
    
    // Boundary conditions for this problem
    Array<int> essential_attr(pmesh->bdr_attributes.Size());
    essential_attr[0] = 1;
    essential_attr[1] = 1;
    
    m = new ParBilinearForm(pfes);
    m->AddDomainIntegrator(new MassIntegrator());
    m->Assemble(skip_zeros);
    m->Finalize(skip_zeros);
    m->EliminateEssentialBC(essential_attr);
    
    k = new ParBilinearForm(pfes);    
    k->AddDomainIntegrator(new DiffusionIntegrator(*(new ConstantCoefficient(-p_[0]))));
    Vector p2(fes->GetParMesh()->SpaceDimension());
    p2 = p_[1];
    k->AddDomainIntegrator(new ConvectionIntegrator(*(new VectorConstantCoefficient(p2))));
    k->Assemble(skip_zeros);
    k->Finalize(skip_zeros);
    k->EliminateEssentialBC(essential_attr);

    M = m->ParallelAssemble();
    K = k->ParallelAssemble();

    M_prec.SetType(HypreSmoother::Jacobi);
    M_solver.SetPreconditioner(M_prec);
    M_solver.SetOperator(*M);

    M_solver.SetRelTol(1e-9);
    M_solver.SetAbsTol(0.0);
    M_solver.SetMaxIter(1000);
    M_solver.SetPrintLevel(0);
    
  }

  virtual void Mult(const Vector &x, Vector &y) const;
  virtual void QuadratureIntegration(const Vector &x, Vector &y) const;
  virtual void AdjointRateMult(const Vector &y, Vector &yB, Vector &yBdot) const;
  virtual void ObjectiveSensitivityMult(const Vector &y, const Vector &yB, Vector &qbdot) const;
  virtual int ImplicitSetupB(const double t, const Vector &y, const Vector &yB,
			     const Vector &fyB, int jokB, int *jcurB, double gammaB);
  virtual int ImplicitSolveB(Vector &x, const Vector &b, double tol);
  
protected:
  Vector p_;

  ParFiniteElementSpace *pfes;
  
  // Internal matrices
  ParBilinearForm * m;
  ParBilinearForm * k;
  HypreParMatrix *M;
  HypreParMatrix *K;

  CGSolver M_solver;
  HypreSmoother M_prec;

  // Solvers
  GMRESSolver adjointSolver;  
  SparseMatrix* adjointMatrix;
};

double u_init(const Vector &x) 
{
  return x[0]*(2. - x[0])*exp(2.*x[0]);
}

int main(int argc, char *argv[])
{
   // 1. Initialize MPI.
   int num_procs, myid;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);
  
   // 1. Parse command-line options.
   int ser_ref_levels = 0;
   int par_ref_levels = 0;
   int order = 3;
   int ode_solver_type = 4;
   double t_final = 2.5;
   double dt = 0.01;
   int mx = 20;
   
   bool visualization = true;
   bool visit = false;
   bool binary = false;
   int vis_steps = 5;

   // Relative and absolute tolerances for CVODES
   double reltol = 1e-8, abstol = 1e-5;

   int precision = 8;
   cout.precision(precision);

   OptionsParser args(argc, argv);

   args.AddOption(&ser_ref_levels, "-r", "--refine",
                  "Number of times to refine the mesh uniformly.");
   args.AddOption(&order, "-o", "--order",
                  "Order (degree) of the finite elements.");
   args.AddOption(&ode_solver_type, "-s", "--ode-solver",
                  "ODE solver: 1 - CVODES (adaptive order) implicit Adams,\n\t"
                  "            2 - ARKODE default (4th order) explicit,\n\t"
                  "            3 - ARKODE RK8.");
   args.AddOption(&t_final, "-tf", "--t-final",
                  "Final time; start time is 0.");
   args.AddOption(&dt, "-dt", "--time-step",
                  "Time step.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&visit, "-visit", "--visit-datafiles", "-no-visit",
                  "--no-visit-datafiles",
                  "Save data files for VisIt (visit.llnl.gov) visualization.");
   args.AddOption(&binary, "-binary", "--binary-datafiles", "-ascii",
                  "--ascii-datafiles",
                  "Use binary (Sidre) or ascii format for VisIt data files.");
   args.AddOption(&vis_steps, "-vs", "--visualization-steps",
                  "Visualize every n-th timestep.");
   
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   // check for vaild ODE solver option
   if (ode_solver_type < 1 || ode_solver_type > 4)
   {
      cout << "Unknown ODE solver type: " << ode_solver_type << '\n';
      return 3;
   }
   args.PrintOptions(cout);

   // 2. Read the mesh from the given mesh file. We can handle geometrically
   //    periodic meshes in this code.
   //   Mesh *mesh = new Mesh(mx+1,1,Element::QUADRILATERAL, true, 2.0, 1.0);
   Mesh *mesh = new Mesh(mx+1, 2.);
   int dim = 2;

   // 3. Refine the mesh to increase the resolution. In this example we do
   //    'ref_levels' of uniform refinement, where 'ref_levels' is a
   //    command-line parameter. If the mesh is of NURBS type, we convert it to
   //    a (piecewise-polynomial) high-order mesh.
   for (int lev = 0; lev < ser_ref_levels; lev++)
   {
      mesh->UniformRefinement();
   }
   if (mesh->NURBSext)
   {
      mesh->SetCurvature(max(order, 1));
   }
   mesh->GetBoundingBox(bb_min, bb_max, max(order, 1));

   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   delete mesh;
   for (int lev = 0; lev < par_ref_levels; lev++)
   {
      pmesh->UniformRefinement();
   }

   //6. Finite Element Spaces

   H1_FECollection fec(1, dim);
   ParFiniteElementSpace *fes = new ParFiniteElementSpace(pmesh, &fec);

   HYPRE_Int global_vSize = fes->GlobalTrueVSize();
   if (myid == 0)
   {
      cout << "Number of unknowns: " << global_vSize << endl;
   }
   
   // 7. Define the time-dependent evolution operator describing the ODE
   //    right-hand side, and define the ODE solver used for time integration.
   
   Vector p(2);
   p[0] = 1.0;
   p[1] = 0.5;

   // u is the size of the solution vector
   ParGridFunction u(fes);
   FunctionCoefficient u0(u_init);
   u.ProjectCoefficient(u0);

   cout << "Init u: " << endl;
   u.Print();
   
   AdvDiffSUNDIALS adv(u.Size(), p, fes);

   
   
   double t = 0.0;
   adv.SetTime(t);

   // Create the time integrator
   ODESolver *ode_solver = NULL;
   CVODESolver *cvode = NULL;
   CVODESSolver *cvodes = NULL;
   ARKStepSolver *arkode = NULL;

   int steps = 200;
   
   switch (ode_solver_type)
     {
     case 4:
       cvodes = new CVODESSolver(CV_ADAMS);
       cvodes->Init(adv, t, u);
       cvodes->SetSStolerances(reltol, abstol);
       //       cvodes->SetMaxStep(dt);
       //       cvodes->InitQuadIntegration(1.e-6,1.e-6);
       cvodes->InitAdjointSolve(steps);
       ode_solver = cvodes; break;
     }

   // 8. Perform time-integration (looping over the time iterations, ti,
   //    with a time-step dt).
   bool done = false;
   for (int ti = 0; !done; )
   {
      double dt_real = max(dt, t_final - t);
      ode_solver->Step(u, t, dt_real);
      ti++;

      done = (t >= t_final - 1e-8*dt);

      if (done || ti % vis_steps == 0)
      {
   	//         cout << "time step: " << ti << ", time: " << t << endl;
         if (cvode) { cvode->PrintInfo(); }
         if (arkode) { arkode->PrintInfo(); }
   	 if (cvodes) { cvodes->PrintInfo(); }

      }
   }
   
   cout << "Final Solution: " << t << endl;
   u.Print();

   // Calculate int_x u dx at t = 5
   if (cvodes) {
     ParLinearForm obj(fes);
     ConstantCoefficient one(1.0);
     obj.AddDomainIntegrator(new DomainLFIntegrator(one));
     obj.Assemble();
     
     double g = obj(u);
     if (myid == 0)
       {
	 cout << "g: " << g << endl;
       }
   }

   // // backward portion
   // Vector w(3);
   // w=0.;
   // double TBout1 = 40.;
   // Vector dG_dp(3);
   // dG_dp=0.;
   // if (cvodes) {
   //   t = t_final;
   //   cvodes->InitB(adv, t, w);
   //   cvodes->InitQuadIntegrationB(1.e-6, 1.e-6);
   //   // Commenting this line back in fails
   //   cvodes->SetLinearSolverB();
     
   //   // Results at time TBout1
   //   double dt_real = max(dt, t - TBout1);
   //   cvodes->StepB(w, t, dt_real);
   //   cvodes->GetCorrespondingForwardSolution(t, u);
   //   cout << "t: " << t << endl;
   //   cout << "w:" << endl;
   //   w.Print();
   //   cout << "u:" << endl;
   //   u.Print();

   //   // Results at T0
   //   dt_real = max(dt, t - 0.);
   //   cvodes->StepB(w, t, dt_real);
   //   cvodes->GetCorrespondingForwardSolution(t, u);
   //   cout << "t: " << t << endl;
   //   cout << "w:" << endl;
   //   w.Print();
   //   cout << "u:" << endl;
   //   u.Print();

   //   // Evaluate Sensitivity
   //   cvodes->EvalObjectiveSensitivity(t, dG_dp);
   //   cout << "dG/dp:" << endl;
   //   dG_dp.Print();
     
   // }
   
   // 10. Free the used memory.
   delete ode_solver;
   
   return 0;
}

// AdvDiff Implementation
void AdvDiffSUNDIALS::Mult(const Vector &x, Vector &y) const
{
  Vector z(x.Size());
  K->Mult(x, z);
  M_solver.Mult(z, y);
}


void AdvDiffSUNDIALS::QuadratureIntegration(const Vector &y, Vector &qdot) const
{
  qdot[0] = y[2];
}


void AdvDiffSUNDIALS::AdjointRateMult(const Vector &y, Vector & yB, Vector &yBdot) const
{
  double l21 = (yB[1]-yB[0]);
  double l32 = (yB[2]-yB[1]);
  double p1 = p_[0];
  double p2 = p_[1];
  double p3 = p_[2];
  yBdot[0] = -p1 * l21;
  yBdot[1] = p2 * y[2] * l21 - 2. * p3 * y[1] * l32;
  yBdot[2] = p2 * y[1] * l21 - 1.0;
}

void AdvDiffSUNDIALS::ObjectiveSensitivityMult(const Vector &y, const Vector &yB, Vector &qBdot) const
{
  double l21 = (yB[1]-yB[0]);
  double l32 = (yB[2]-yB[1]);
  double y23 = y[1] * y[2];

  qBdot[0] = y[0] * l21;
  qBdot[1] = -y23 * l21;
  qBdot[2] = y[1]*y[1]*l32;
}

int AdvDiffSUNDIALS::ImplicitSetupB(const double t, const Vector &y, const Vector &yB,
				    const Vector &fyB, int jokB, int *jcurB, double gammaB)
{

  // M = I- gamma J
  // J = dfB/dyB
  // fB
  // Let's create a SparseMatrix and fill in the entries since this example doesn't contain finite elements
  
  delete adjointMatrix;
  adjointMatrix = new SparseMatrix(y.Size(), yB.Size());
  for (int j = 0; j < y.Size(); j++)
    {
      Vector JacBj(yB.Size());
      Vector yBone(yB.Size());
      yBone = 0.;
      yBone[j] = 1.;
      AdjointRateMult(y, yBone, JacBj);
      JacBj[2] += 1.;
      for (int i = 0; i < y.Size(); i++) {
	adjointMatrix->Set(i,j, (i == j ? 1.0 : 0.) - gammaB * JacBj[i]);	
      }
    }

  *jcurB = 1;
  adjointMatrix->Finalize();
  //  adjointMatrix->PrintMatlab();
  //  y.Print();
  adjointSolver.SetOperator(*adjointMatrix);
  
  return 0;
}

// Is b = -fB ?
// is tol reltol or abstol?
int AdvDiffSUNDIALS::ImplicitSolveB(Vector &x, const Vector &b, double tol)
{
  adjointSolver.SetRelTol(1e-14);
  adjointSolver.Mult(b, x);
  return(0);
}
