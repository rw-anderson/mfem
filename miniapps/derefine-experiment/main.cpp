#include "mfem.hpp"

#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

const char *vishost = "localhost";
const int visport = 19916;
const int visw = 480;
const int vish = 480;
const int border = 15;
socketstream vis;
bool visualization = true;

void vismesh(Mesh* mesh)
{
   if (visualization && vis.good())
   {
      vis.precision(8);
      vis << "mesh" << endl << *mesh << flush;
      vis << "window_title '" << "Mesh" << "'" << endl
          << "window_geometry "
          << (vish + border) << " " << 0
          << " " << visw << " " << vish  << endl
          << "keys mgeA" << endl;
   }
}

int main(int argc, char *argv[])
{
   vis.open(vishost, visport);

   const char *mesh_file = "../../data/star.mesh";
   
   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   mesh->EnsureNCMesh();

   Array<Refinement> refs(1);
   refs[0] = Refinement(0);
   mesh->GeneralRefinement(refs, 1, 1);

   vismesh(mesh);
   
   return 0;
}
