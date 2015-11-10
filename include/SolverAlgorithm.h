/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


#ifndef SolverAlgorithm_h
#define SolverAlgorithm_h

#include <Algorithm.h>

#include <stk_mesh/base/Entity.hpp>
#include <vector>
#include <Kokkos_Core.hpp>

namespace sierra{
namespace nalu{

class EquationSystem;
class Realm;

class SolverAlgorithm : public Algorithm
{
public:

  SolverAlgorithm(
    Realm &realm,
    stk::mesh::Part *part,
    EquationSystem *eqSystem);
  virtual ~SolverAlgorithm() {}

  virtual void execute() = 0;
  virtual void initialize_connectivity() = 0;

protected:

  // Need to find out whether this ever gets called inside a modification cycle.
  void apply_coeff(
    const std::vector<stk::mesh::Entity> & sym_meshobj,
    const std::vector<double> &rhs,
    const std::vector<double> &lhs,
    const char *trace_tag=0);
  
  /*void apply_coeff(
      const Kokkos::View<const stk::mesh::Entity*> & sym_meshobj,
      const Kokkos::View<const double*> & rhs,
      const Kokkos::View<const double*> & lhs, const char *trace_tag=0);*/

  void apply_coeff(
      const Kokkos::View<const stk::mesh::Entity*, Kokkos::DefaultExecutionSpace::scratch_memory_space, Kokkos::MemoryUnmanaged> & sym_meshobj,
      const Kokkos::View<const double*, Kokkos::DefaultExecutionSpace::scratch_memory_space, Kokkos::MemoryUnmanaged> & rhs,
      const Kokkos::View<const double*, Kokkos::DefaultExecutionSpace::scratch_memory_space, Kokkos::MemoryUnmanaged> & lhs, const char *trace_tag=0);
  EquationSystem *eqSystem_;
};

} // namespace nalu
} // namespace Sierra

#endif
