/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


// nalu
#include <AssembleElemSolverAlgorithm.h>
#include <EquationSystem.h>
#include <SolverAlgorithm.h>
#include <master_element/MasterElement.h>

#include <FieldTypeDef.h>
#include <KokkosInterface.h>
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

namespace sierra{
namespace nalu{

//==========================================================================
// Class Definition
//==========================================================================
// AssembleElemSolverAlgorithm - add LHS/RHS for element-based contribution
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
AssembleElemSolverAlgorithm::AssembleElemSolverAlgorithm(
  Realm &realm,
  stk::mesh::Part *part,
  EquationSystem *eqSystem)
  : SolverAlgorithm(realm, part, eqSystem),
    sizeOfSystem_(eqSystem->linsys_->numDof())
{
  // nothing
}

//--------------------------------------------------------------------------
//-------- initialize_connectivity -----------------------------------------
//--------------------------------------------------------------------------
void
AssembleElemSolverAlgorithm::initialize_connectivity()
{
  eqSystem_->linsys_->buildElemToNodeGraph(partVec_);
}

//--------------------------------------------------------------------------
//-------- execute ---------------------------------------------------------
//--------------------------------------------------------------------------
void
AssembleElemSolverAlgorithm::execute()
{
  stk::mesh::MetaData & meta_data = realm_.meta_data();

  if(supplementalAlg_.empty()) return;

  const int numWorkBuckets = 32;
  const int maxNodesPerElement = 8;
  const int maxElementsPerBucket = 512;
  const int lhsSize = maxNodesPerElement*maxNodesPerElement*sizeOfSystem_*sizeOfSystem_;
  const int rhsSize = maxNodesPerElement*sizeOfSystem_;

  // supplemental algorithm size and setup
  const size_t supplementalAlgSize = supplementalAlg_.size();
  for ( size_t i = 0; i < supplementalAlgSize; ++i )
    supplementalAlg_[i]->setup();

  // define some common selectors
  stk::mesh::Selector s_locally_owned_union = meta_data.locally_owned_part()
    &stk::mesh::selectUnion(partVec_);

  stk::mesh::BucketVector const& elem_buckets =
    realm_.get_buckets( stk::topology::ELEMENT_RANK, s_locally_owned_union );

  const int bytes_per_team = 0;
  const int bytes_per_thread =
      SharedMemView<double *>::shmem_size(lhsSize) +
      SharedMemView<double *>::shmem_size(rhsSize) +
      SharedMemView<stk::mesh::Entity *>::shmem_size(maxNodesPerElement) +
      SharedMemView<int *>::shmem_size(rhsSize);
  auto team_exec = get_team_policy(elem_buckets.size(), bytes_per_team, bytes_per_thread);

  Kokkos::parallel_for("Nalu::AssembleElemSolverAlgorithm::execute",
     team_exec, [&] (const DeviceTeam & team) {
    const int ib = team.league_rank();
    const stk::mesh::Bucket & b = *elem_buckets[ib];
    const stk::mesh::Bucket::size_type length   = b.size();

    // extract master element
    MasterElement *meSCS = realm_.get_surface_master_element(b.topology());
    MasterElement *meSCV = realm_.get_volume_master_element(b.topology());

    // extract master element specifics
    const int nodesPerElement = meSCS->nodesPerElement_;

    Kokkos::single(Kokkos::PerTeam(team), [&] (){
      // resize possible supplemental element alg
      for ( size_t i = 0; i < supplementalAlgSize; ++i )
        supplementalAlg_[i]->elem_resize(meSCS, meSCV);
    });
    team.team_barrier();

    SharedMemView<stk::mesh::Entity*> connected_nodes_;
    SharedMemView<double*> lhs_;
    SharedMemView<double*> rhs_;
    SharedMemView<int*> localIdsScratch;
    {
      connected_nodes_ = Kokkos::subview(
          SharedMemView<stk::mesh::Entity**> (team.team_shmem(), team.team_size(), nodesPerElement),
          team.team_rank(), Kokkos::ALL());
      lhs_ = Kokkos::subview(
          SharedMemView<double**>(team.team_shmem(), team.team_size(), lhsSize),
          team.team_rank(), Kokkos::ALL());
      rhs_ = Kokkos::subview(
          SharedMemView<double**>(team.team_shmem(), team.team_size(), rhsSize),
          team.team_rank(), Kokkos::ALL());
      localIdsScratch = Kokkos::subview(
          SharedMemView<int**> (team.team_shmem(), team.team_size(), rhsSize),
          team.team_rank(), Kokkos::ALL());
    }

    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, length), [&] (const size_t k) {

      // get element
      stk::mesh::Entity element = b[k];

      // extract node relations and provide connected nodes
      stk::mesh::Entity const * node_rels = b.begin_nodes(k);
      int num_nodes = b.num_nodes(k);

      // sanity check on num nodes
//        ThrowAssert( num_nodes == nodesPerElement ); throws are bad

      for ( int ni = 0; ni < num_nodes; ++ni ) {
        stk::mesh::Entity node = node_rels[ni];
        // set connected nodes
        connected_nodes_(ni) = node;
      }

      for ( int i = 0; i < lhsSize; ++i )
        lhs_(i) = 0.0;
      for ( int i = 0; i < rhsSize; ++i )
        rhs_(i) = 0.0;

      // call supplemental; gathers happen inside the elem_execute method
      for ( size_t i = 0; i < supplementalAlgSize; ++i )
        supplementalAlg_[i]->elem_execute(&lhs_(0), &rhs_(0), element, meSCS, meSCV);

      apply_coeff(connected_nodes_, rhs_, lhs_, localIdsScratch,  __FILE__);

    });
  });
}

} // namespace nalu
} // namespace Sierra
