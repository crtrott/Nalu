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

#include<Kokkos_Core.hpp>
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
//  Kokkos::View<double******> lhs("lhsScratch", numWorkBuckets, maxElementsPerBucket,
//      maxNodesPerElement, maxNodesPerElement, sizeOfSystem_, sizeOfSystem_);
  Kokkos::View<double***, Kokkos::LayoutRight> lhsScratch("lhsScratch", numWorkBuckets, maxElementsPerBucket, lhsSize);
  Kokkos::View<double***, Kokkos::LayoutRight> rhsScratch("rhsScratch", numWorkBuckets, maxElementsPerBucket, rhsSize);
  Kokkos::View<stk::mesh::Entity***, Kokkos::LayoutRight> connectedNodesScratch("cnScratch", numWorkBuckets, maxElementsPerBucket, maxNodesPerElement);


  // supplemental algorithm size and setup
  const size_t supplementalAlgSize = supplementalAlg_.size();
  for ( size_t i = 0; i < supplementalAlgSize; ++i )
    supplementalAlg_[i]->setup();

  // define some common selectors
  stk::mesh::Selector s_locally_owned_union = meta_data.locally_owned_part()
    &stk::mesh::selectUnion(partVec_);

  typedef typename Kokkos::TeamPolicy<Kokkos::Serial>::member_type team_type;
  stk::mesh::BucketVector const& elem_buckets =
    realm_.get_buckets( stk::topology::ELEMENT_RANK, s_locally_owned_union );
  for (unsigned bucketOffset = 0; bucketOffset < elem_buckets.size(); bucketOffset += numWorkBuckets)
  {
    const int bucketEnd = std::min(bucketOffset + numWorkBuckets, (unsigned int)elem_buckets.size());
    Kokkos::parallel_for("Nalu::AssembleElemSolverAlgorithm::execute",
        Kokkos::TeamPolicy<Kokkos::Serial>(bucketEnd-bucketOffset, Kokkos::AUTO), [&] (const team_type& team) {
      const int ib = team.league_rank();
      const stk::mesh::Bucket & b = *elem_buckets[bucketOffset+ib];
      const stk::mesh::Bucket::size_type length   = b.size();

      // extract master element
      MasterElement *meSCS = realm_.get_surface_master_element(b.topology());
      MasterElement *meSCV = realm_.get_volume_master_element(b.topology());

      // extract master element specifics
      const int nodesPerElement = meSCS->nodesPerElement_;

      // KOKKOS this could be a critical region so that each team only executes this once instead of every thread
      // TODO fix me travis/victor
      Kokkos::single(Kokkos::PerTeam(team), [&] (){
      // resize possible supplemental element alg
      for ( size_t i = 0; i < supplementalAlgSize; ++i )
        supplementalAlg_[i]->elem_resize(meSCS, meSCV);
      });
      team.team_barrier();

      Kokkos::parallel_for(Kokkos::TeamThreadRange(team, length), [&] (const size_t k) {

        // get element
        stk::mesh::Entity element = b[k];

        double * const p_lhs = &lhsScratch(ib,k,0);
        double * const p_rhs = &rhsScratch(ib,k,0);

        stk::mesh::Entity * p_connected_nodes = &connectedNodesScratch(ib,k,0);
        // extract node relations and provide connected nodes
        stk::mesh::Entity const * node_rels = b.begin_nodes(k);
        int num_nodes = b.num_nodes(k);

        // sanity check on num nodes
//        ThrowAssert( num_nodes == nodesPerElement ); throws are bad

        for ( int ni = 0; ni < num_nodes; ++ni ) {
          stk::mesh::Entity node = node_rels[ni];
          // set connected nodes
          p_connected_nodes[ni] = node;
        }

        for ( int i = 0; i < lhsSize; ++i )
          p_lhs[i] = 0.0;
        for ( int i = 0; i < rhsSize; ++i )
          p_rhs[i] = 0.0;

        // call supplemental; gathers happen inside the elem_execute method
        for ( size_t i = 0; i < supplementalAlgSize; ++i )
          supplementalAlg_[i]->elem_execute( p_lhs, p_rhs, element, meSCS, meSCV);

        /*auto connected_nodes = Kokkos::subview(connectedNodesScratch,ib,k,Kokkos::ALL());
        auto rhs = Kokkos::subview(rhsScratch,ib,k,Kokkos::ALL());
        auto lhs = Kokkos::subview(lhsScratch,ib,k,Kokkos::ALL());
        apply_coeff(connected_nodes, rhs, lhs, __FILE__);*/

      });
    });
  }
}

} // namespace nalu
} // namespace Sierra
