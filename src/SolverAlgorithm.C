/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


#include <SolverAlgorithm.h>
#include <Algorithm.h>
#include <EquationSystem.h>
#include <LinearSystem.h>

#include <stk_mesh/base/Entity.hpp>

#include <vector>

namespace sierra{
namespace nalu{

class Realm;
class EquationSystem;

//==========================================================================
// Class Definition
//==========================================================================
// SolverAlgorithm - base class for algorithm with expectations of solver
//                   contributions
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
SolverAlgorithm::SolverAlgorithm(
  Realm &realm,
  stk::mesh::Part *part,
  EquationSystem *eqSystem)
  : Algorithm(realm, part),
    eqSystem_(eqSystem)
{
  // does nothing
}

//--------------------------------------------------------------------------
//-------- apply_coeff -----------------------------------------------------
//--------------------------------------------------------------------------
void
SolverAlgorithm::apply_coeff(
  const std::vector<stk::mesh::Entity> & sym_meshobj,
  const std::vector<double> & rhs,
  const std::vector<double> & lhs, const char *trace_tag)
{
  eqSystem_->linsys_->sumInto(sym_meshobj, rhs, lhs, trace_tag);
}

/*
void
SolverAlgorithm::apply_coeff(
    const Kokkos::View<const stk::mesh::Entity*> & sym_meshobj,
    const Kokkos::View<const double*> & rhs,
    const Kokkos::View<const double*> & lhs, const char *trace_tag)
{
  eqSystem_->linsys_->sumInto(sym_meshobj, rhs, lhs, trace_tag);
}
*/
void
SolverAlgorithm::apply_coeff(
    const SharedMemView<const stk::mesh::Entity*> & entities,
    const SharedMemView<const double*> & rhs,
    const SharedMemView<const double*> & lhs,
    const SharedMemView<int*> & localIds,
    const char *trace_tag)
{
  eqSystem_->linsys_->sumInto(entities, rhs, lhs, localIds, trace_tag);
}

} // namespace nalu
} // namespace Sierra
