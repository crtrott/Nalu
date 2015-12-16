/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


// nalu
#include <AssembleScalarEdgeDiffSolverAlgorithm.h>
#include <EquationSystem.h>
#include <FieldTypeDef.h>
#include <KokkosInterface.h>
#include <LinearSystem.h>
#include <Realm.h>

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
// AssembleScalarEdgeDiffSolverAlgorithm - add LHS/RHS for scalar diffusion
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
AssembleScalarEdgeDiffSolverAlgorithm::AssembleScalarEdgeDiffSolverAlgorithm(
  Realm &realm,
  stk::mesh::Part *part,
  EquationSystem *eqSystem,
  ScalarFieldType *scalarQ,
  VectorFieldType *dqdx,
  ScalarFieldType *diffFluxCoeff)
  : SolverAlgorithm(realm, part, eqSystem),
    scalarQ_(scalarQ),
    dqdx_(dqdx),
    diffFluxCoeff_(diffFluxCoeff)
{
  // save off fields
  stk::mesh::MetaData & meta_data = realm_.meta_data();
  coordinates_ = meta_data.get_field<VectorFieldType>(stk::topology::NODE_RANK, realm_.get_coordinates_name());
  edgeAreaVec_ = meta_data.get_field<VectorFieldType>(stk::topology::EDGE_RANK, "edge_area_vector");
}

void
AssembleScalarEdgeDiffSolverAlgorithm::initialize_connectivity()
{
  eqSystem_->linsys_->buildEdgeToNodeGraph(partVec_);
}

//--------------------------------------------------------------------------
//-------- execute ---------------------------------------------------------
//--------------------------------------------------------------------------
void
AssembleScalarEdgeDiffSolverAlgorithm::execute()
{

  stk::mesh::BulkData & bulk_data = realm_.bulk_data();
  stk::mesh::MetaData & meta_data = realm_.meta_data();

  const int nDim = meta_data.spatial_dimension();

  // deal with state
  ScalarFieldType &scalarQNp1  = scalarQ_->field_of_state(stk::mesh::StateNP1);

  // define some common selectors
  stk::mesh::Selector s_locally_owned_union = meta_data.locally_owned_part()
    & stk::mesh::selectUnion(partVec_) 
    & !(realm_.get_inactive_selector());

  stk::mesh::BucketVector const& edge_buckets =
    realm_.get_buckets( stk::topology::EDGE_RANK, s_locally_owned_union );

  // space for LHS/RHS; always edge connectivity
  const int nodesPerEdge = 2;
  const int lhsSize = nodesPerEdge*nodesPerEdge;
  const int rhsSize = nodesPerEdge;

  const int bytes_per_team = 0;
  const int bytes_per_thread = SharedMemView<double*>::shmem_size(lhsSize)
      + SharedMemView<double*>::shmem_size(rhsSize)
      + SharedMemView<int*>::shmem_size(nodesPerEdge) // local ID scratch
      + SharedMemView<stk::mesh::Entity*>::shmem_size(nodesPerEdge);

  auto team_exec = get_team_policy(edge_buckets.size(), bytes_per_team, bytes_per_thread);
  Kokkos::parallel_for("Nalu::AssembleScalarEdgeDiffSolverAlgorithm",
      team_exec, [&](const DeviceTeam & team) {
    const int ib = team.league_rank();
    stk::mesh::Bucket & b = *edge_buckets[ib];
    const stk::mesh::Bucket::size_type length   = b.size();

    // TODO: Should be per-thread
    SharedMemView<double*> lhs(team.team_shmem(), lhsSize);
    SharedMemView<double*> rhs(team.team_shmem(), rhsSize);
    SharedMemView<stk::mesh::Entity*> connected_nodes(team.team_shmem(), nodesPerEdge);
    SharedMemView<int*> localIdsScratch(team.team_shmem(), nodesPerEdge);

    // pointer to edge area vector and mdot
    Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::MemoryUnmanaged>
      av(stk::mesh::field_data(*edgeAreaVec_, b), length, nDim);

    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, length), [&] (const size_t k) {

      // zeroing of lhs/rhs
      for ( int i = 0; i < lhsSize; ++i ) {
        lhs[i] = 0.0;
      }
      for ( int i = 0; i < rhsSize; ++i ) {
        rhs[i] = 0.0;
      }

      // get edge
      stk::mesh::Entity edge = b[k];
      stk::mesh::Entity const* edge_node_rels = bulk_data.begin_nodes(edge);

      // sanity check on number or nodes
      //ThrowAssert( bulk_data.num_nodes(edge) == 2 );

      // pointer to edge area vector
      auto p_areaVec = Kokkos::subview(av, k, Kokkos::ALL());

      // left and right nodes
      stk::mesh::Entity nodeL = edge_node_rels[0];
      stk::mesh::Entity nodeR = edge_node_rels[1];

      connected_nodes[0] = nodeL;
      connected_nodes[1] = nodeR;

      // extract nodal fields
      double * coordL = stk::mesh::field_data( *coordinates_, nodeL );
      double * coordR = stk::mesh::field_data( *coordinates_, nodeR );

      double * dqdxL = stk::mesh::field_data( *dqdx_, nodeL );
      double * dqdxR = stk::mesh::field_data( *dqdx_, nodeR );

      double qNp1L = *stk::mesh::field_data( scalarQNp1, nodeL );
      double qNp1R = *stk::mesh::field_data( scalarQNp1, nodeR );

      double diffFluxCoeffL = *stk::mesh::field_data( *diffFluxCoeff_, nodeL );
      double diffFluxCoeffR = *stk::mesh::field_data( *diffFluxCoeff_, nodeR );

      // ip props
      const double viscIp = 0.5*(diffFluxCoeffL + diffFluxCoeffR);

      // compute geometry
      double axdx = 0.0;
      double asq = 0.0;
      for ( int j = 0; j < nDim; ++j ) {
        const double axj = p_areaVec[j];
        const double dxj = coordR[j] - coordL[j];
        asq += axj*axj;
        axdx += axj*dxj;
      }

      const double inv_axdx = 1.0/axdx;

      // NOC
      double nonOrth = 0.0;
      for ( int j = 0; j < nDim; ++j ) {
        const double axj = p_areaVec[j];
        const double dxj = coordR[j] - coordL[j];
        // now non-orth (over-relaxed procedure of Jasek)
        const double kxj = axj - asq*inv_axdx*dxj;
        const double GjIp = 0.5*(dqdxL[j] + dqdxR[j]);
        nonOrth += -viscIp*kxj*GjIp;
      }

      //====================================
      // diffusive flux
      //====================================
      double lhsfac = -viscIp*asq*inv_axdx;
      double diffFlux = lhsfac*(qNp1R - qNp1L) + nonOrth;

      // first left
      lhs[0] = -lhsfac;
      lhs[1] = +lhsfac;
      rhs[0] = -diffFlux;

      // now right
      lhs[2] = +lhsfac;
      lhs[3] = -lhsfac;
      rhs[1] = diffFlux;

      // apply it
      apply_coeff(connected_nodes, rhs, lhs, localIdsScratch, __FILE__);

    });
  });
}

} // namespace nalu
} // namespace Sierra
