/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


// nalu
#include <AssembleNodalGradBoundaryAlgorithm.h>
#include <Algorithm.h>

#include <FieldTypeDef.h>
#include <Realm.h>
#include <TimeIntegrator.h>
#include <master_element/MasterElement.h>

// stk_mesh/base/fem
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Field.hpp>
#include <stk_mesh/base/GetBuckets.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/Part.hpp>

#include <KokkosInterface.h>

namespace sierra{
namespace nalu{

//==========================================================================
// Class Definition
//==========================================================================
// AssembleNodalGradBoundaryAlgorithm - adds in boundary contribution
//                                      for elem/edge proj nodal gradient
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
AssembleNodalGradBoundaryAlgorithm::AssembleNodalGradBoundaryAlgorithm(
  Realm &realm,
  stk::mesh::Part *part,
  ScalarFieldType *scalarQ,
  VectorFieldType *dqdx,
  const bool useShifted)
  : Algorithm(realm, part),
    scalarQ_(scalarQ),
    dqdx_(dqdx),
    useShifted_(useShifted)
{
  // nothing to do
}

//--------------------------------------------------------------------------
//-------- execute ---------------------------------------------------------
//--------------------------------------------------------------------------
void
AssembleNodalGradBoundaryAlgorithm::execute()
{

  stk::mesh::MetaData & meta_data = realm_.meta_data();

  const int nDim = meta_data.spatial_dimension();

  // extract fields
  GenericFieldType *exposedAreaVec = meta_data.get_field<GenericFieldType>(meta_data.side_rank(), "exposed_area_vector");
  ScalarFieldType *dualNodalVolume = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "dual_nodal_volume");

  // define some common selectors
  stk::mesh::Selector s_locally_owned_union = meta_data.locally_owned_part()
    &stk::mesh::selectUnion(partVec_);

  stk::mesh::BucketVector const& face_buckets =
    realm_.get_buckets( meta_data.side_rank(), s_locally_owned_union );

  const int maxNodesPerFace = 4;
  const int maxNumScsIp = 8;
  const int bytes_per_team = SharedMemView<double **>::shmem_size(maxNumScsIp, maxNodesPerFace);
  const int bytes_per_thread = SharedMemView<double *>::shmem_size(maxNodesPerFace);

  auto team_exec = get_team_policy(face_buckets.size(), bytes_per_team, bytes_per_thread);

  Kokkos::parallel_for("AssembleNodalGradBoundaryAlgorithm",
      team_exec, [&](const DeviceTeam & team) {
    const stk::mesh::Bucket & b = *face_buckets[team.league_rank()];
    const stk::mesh::Bucket::size_type length = b.size();

    // extract master element
    MasterElement *meFC = realm_.get_surface_master_element(b.topology());

    // extract master element specifics
    const int nodesPerFace = meFC->nodesPerElement_;
    const int numScsIp = meFC->numIntPoints_;
    const int *ipNodeMap = meFC->ipNodeMap();

    // algorithm related
    const int scratch_level = 2;
    SharedMemView<double *> ws_scalarQ(team.thread_scratch(scratch_level), nodesPerFace);
    SharedMemView<double **> ws_shape_function(team.team_scratch(scratch_level), numScsIp, nodesPerFace);

    Kokkos::single(Kokkos::PerTeam(team), [&]() {
      if ( useShifted_ )
        meFC->shifted_shape_fcn(&ws_shape_function(0, 0));
      else
        meFC->shape_fcn(&ws_shape_function(0, 0));
    });
    team.team_barrier();

    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, length), [&](const size_t k) {
      // face data
      const double * areaVec = stk::mesh::field_data(*exposedAreaVec, b, k);

      //===============================================
      // gather nodal data; this is how we do it now..
      //===============================================
      stk::mesh::Entity const * face_node_rels = b.begin_nodes(k);
      int num_nodes = b.num_nodes(k);

      // sanity check on num nodes
      ThrowAssert( num_nodes == nodesPerFace );

      for ( int ni = 0; ni < num_nodes; ++ni ) {
        stk::mesh::Entity node = face_node_rels[ni];
        // gather scalars
        ws_scalarQ[ni] = *stk::mesh::field_data(*scalarQ_, node);
      }

      // start assembly
      for ( int ip = 0; ip < numScsIp; ++ip ) {

        // nearest node
        const int nn = ipNodeMap[ip];

        stk::mesh::Entity nodeNN = face_node_rels[nn];

        // pointer to fields to assemble
        double *gradQNN = stk::mesh::field_data(*dqdx_, nodeNN);

        // suplemental
        double volNN = *stk::mesh::field_data(*dualNodalVolume, nodeNN);

        // interpolate to scs point; operate on saved off ws_field
        double qIp = 0.0;
        const int offSet = ip*nodesPerFace;
        for ( int ic = 0; ic < nodesPerFace; ++ic ) {
          qIp += ws_shape_function(ip, ic)*ws_scalarQ[ic];
        }

        // nearest node volume
        double inv_volNN = 1.0/volNN;

        // assemble to nearest node
        for ( int j = 0; j < nDim; ++j ) {
          double fac = qIp*areaVec[ip*nDim+j];
          Kokkos::atomic_add(&gradQNN[j], fac*inv_volNN);
        }
      }
    });
  });
}

} // namespace nalu
} // namespace Sierra
