/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


// nalu
#include <AssembleNodeSolverAlgorithm.h>
#include <EquationSystem.h>
#include <SolverAlgorithm.h>

#include <FieldTypeDef.h>
#include <LinearSystem.h>
#include <Realm.h>
#include <SupplementalAlgorithm.h>
#include <TimeIntegrator.h>

// stk_mesh/base/fem
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Field.hpp>
#include <stk_mesh/base/GetBuckets.hpp>
#include <stk_mesh/base/GetEntities.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/Part.hpp>

#include <KokkosInterface.h>

namespace sierra{
namespace nalu{

//==========================================================================
// Class Definition
//==========================================================================
// AssembleNodeSolverAlgorithm - add LHS/RHS for node-based contribution
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
AssembleNodeSolverAlgorithm::AssembleNodeSolverAlgorithm(
  Realm &realm,
  stk::mesh::Part *part,
  EquationSystem *eqSystem)
  : SolverAlgorithm(realm, part, eqSystem),
    sizeOfSystem_(eqSystem->linsys_->numDof())
{
  // nothing
}

void
AssembleNodeSolverAlgorithm::initialize_connectivity()
{
  eqSystem_->linsys_->buildNodeGraph(partVec_);
}
//--------------------------------------------------------------------------
//-------- execute ---------------------------------------------------------
//--------------------------------------------------------------------------
void
AssembleNodeSolverAlgorithm::execute()
{
  stk::mesh::MetaData & meta_data = realm_.meta_data();

  // space for LHS/RHS
  const int lhsSize = sizeOfSystem_*sizeOfSystem_;
  const int rhsSize = sizeOfSystem_;

  // supplemental algorithm size and setup
  const size_t supplementalAlgSize = supplementalAlg_.size();
  for ( size_t i = 0; i < supplementalAlgSize; ++i )
    supplementalAlg_[i]->setup();

  // define some common selectors
  stk::mesh::Selector s_locally_owned_union = meta_data.locally_owned_part()
    & stk::mesh::selectUnion(partVec_) 
    & !(stk::mesh::selectUnion(realm_.get_slave_part_vector()))
    & !(realm_.get_inactive_selector());

  stk::mesh::BucketVector const& node_buckets =
    realm_.get_buckets( stk::topology::NODE_RANK, s_locally_owned_union );

  const int bytes_per_team = 0;
  const int bytes_per_thread = SharedMemView<double*>::shmem_size(lhsSize)
                             + SharedMemView<double*>::shmem_size(rhsSize)
                             + SharedMemView<stk::mesh::Entity*>::shmem_size(1) // connected nodes
                             + SharedMemView<int*>::shmem_size(rhsSize); // local ID scratch,

  auto team_exec = get_team_policy(node_buckets.size(), bytes_per_team, bytes_per_thread);

  Kokkos::parallel_for("Nalu::AssembleNodeSolver",
      team_exec, [&] (const DeviceTeam & team) {
    const int ib = team.league_rank();
    const stk::mesh::Bucket & b = *node_buckets[ib];
    const stk::mesh::Bucket::size_type length   = b.size();

    SharedMemView<double*> lhs;
    SharedMemView<double*> rhs;
    SharedMemView<stk::mesh::Entity*> connected_nodes;
    SharedMemView<int*> localIdsScratch;
    {
      lhs = Kokkos::subview(
          SharedMemView<double**> (team.team_shmem(), team.team_size(), lhsSize),
          team.team_rank(), Kokkos::ALL());
      rhs = Kokkos::subview(
          SharedMemView<double**> (team.team_shmem(), team.team_size(), rhsSize),
          team.team_rank(), Kokkos::ALL());
      connected_nodes = Kokkos::subview(
          SharedMemView<stk::mesh::Entity**> (team.team_shmem(), team.team_size(), 1),
          team.team_rank(), Kokkos::ALL());
      localIdsScratch = Kokkos::subview(
          SharedMemView<int**> (team.team_shmem(), team.team_size(), rhsSize),
          team.team_rank(), Kokkos::ALL());
    }

    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, length), [&] (const size_t k) {
      // get node
      stk::mesh::Entity node = b[k];
      connected_nodes[0] = node;

      for ( int i = 0; i < lhsSize; ++i )
        lhs[i] = 0.0;
      for ( int i = 0; i < rhsSize; ++i )
        rhs[i] = 0.0;

      // call supplemental
      for ( size_t i = 0; i < supplementalAlgSize; ++i )
        supplementalAlg_[i]->node_execute( &lhs[0], &rhs[0], node);

      apply_coeff(connected_nodes, rhs, lhs, localIdsScratch, __FILE__);

    });
  });
}

} // namespace nalu
} // namespace Sierra
