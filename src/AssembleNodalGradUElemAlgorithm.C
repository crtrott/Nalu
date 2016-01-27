/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


// nalu
#include <AssembleNodalGradUElemAlgorithm.h>
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
// AssembleNodalGradUElemAlgorithm - Green-Gauss gradient
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
AssembleNodalGradUElemAlgorithm::AssembleNodalGradUElemAlgorithm(
  Realm &realm,
  stk::mesh::Part *part,
  VectorFieldType *vectorQ,
  GenericFieldType *dqdx,
  const bool useShifted)
  : Algorithm(realm, part),
    vectorQ_(vectorQ),
    dqdx_(dqdx),
    useShifted_(useShifted)
{
  // nothing to do
}

//--------------------------------------------------------------------------
//-------- execute ---------------------------------------------------------
//--------------------------------------------------------------------------
void
AssembleNodalGradUElemAlgorithm::execute()
{

  stk::mesh::MetaData & meta_data = realm_.meta_data();

  const int nDim = meta_data.spatial_dimension();

  // extract fields
  ScalarFieldType *dualNodalVolume = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "dual_nodal_volume");
  VectorFieldType *coordinates = meta_data.get_field<VectorFieldType>(stk::topology::NODE_RANK, realm_.get_coordinates_name());

  // define some common selectors
  stk::mesh::Selector s_locally_owned_union = meta_data.locally_owned_part()
    & stk::mesh::selectUnion(partVec_)  
    & !(realm_.get_inactive_selector());

  stk::mesh::BucketVector const& elem_buckets =
    realm_.get_buckets( stk::topology::ELEMENT_RANK, s_locally_owned_union );

  const int maxNodesPerElement = 8;
  const int maxNumScsIp = 16;

  const int bytes_per_team = SharedMemView<double **>::shmem_size(maxNumScsIp, maxNodesPerElement);

  const int bytes_per_thread =
      SharedMemView<double **>::shmem_size(maxNodesPerElement, nDim) +
      SharedMemView<double *>::shmem_size(maxNodesPerElement) +
      SharedMemView<double **>::shmem_size(maxNodesPerElement, nDim) +
      SharedMemView<double **>::shmem_size(maxNumScsIp, nDim) +
      SharedMemView<double *>::shmem_size(nDim);

  auto team_exec = get_team_policy(elem_buckets.size(), bytes_per_team, bytes_per_thread);

  Kokkos::parallel_for("AssembleNodalGradUElemAlgorithm::execute",
      team_exec, [&](const DeviceTeam & team) {
    stk::mesh::Bucket & b = *elem_buckets[team.league_rank()];
    const stk::mesh::Bucket::size_type length = b.size();

    // extract master element
    MasterElement *meSCS = realm_.get_surface_master_element(b.topology());

    // extract master element specifics
    const int nodesPerElement = meSCS->nodesPerElement_;
    const int numScsIp = meSCS->numIntPoints_;
    const int *lrscv = meSCS->adjacentNodes();

    // algorithm related
    const int scratch_level = 2;
    SharedMemView<double **> ws_vectorQ(team.thread_scratch(scratch_level), nodesPerElement, nDim);
    SharedMemView<double *> ws_dualVolume(team.thread_scratch(scratch_level), nodesPerElement);
    SharedMemView<double **> ws_coordinates(team.thread_scratch(scratch_level), nodesPerElement, nDim);
    SharedMemView<double **> ws_scs_areav(team.thread_scratch(scratch_level), numScsIp, nDim);
    SharedMemView<double *> qIp(team.thread_scratch(scratch_level), nDim);

    SharedMemView<double **> ws_shape_function(team.team_scratch(scratch_level), numScsIp, nodesPerElement);

    Kokkos::single(Kokkos::PerTeam(team), [&]() {
      if ( useShifted_ )
        meSCS->shifted_shape_fcn(&ws_shape_function(0, 0));
      else
        meSCS->shape_fcn(&ws_shape_function(0, 0));
    });
    team.team_barrier();

    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, length), [&](const size_t k) {
      //===============================================
      // gather nodal data; this is how we do it now..
      //===============================================
      stk::mesh::Entity const * node_rels = b.begin_nodes(k);
      int num_nodes = b.num_nodes(k);

      // sanity check on num nodes
      ThrowAssert( num_nodes == nodesPerElement );

      // note: we absolutely need to gather coords since it
      // is required to compute the area vector. however,
      // ws_scalarQ and ws_dualVolume are choices to avoid
      // field data call for interpolation

      for ( int ni = 0; ni < num_nodes; ++ni ) {
        stk::mesh::Entity node = node_rels[ni];

        // pointers to real data
        double * coords = stk::mesh::field_data(*coordinates, node);
        double * vectorQ = stk::mesh::field_data(*vectorQ_, node);

        // gather scalars
        ws_dualVolume[ni] = *stk::mesh::field_data(*dualNodalVolume, node);

        // gather vectors
        for ( int j=0; j < nDim; ++j ) {
          ws_coordinates(ni, j) = coords[j];
          ws_vectorQ(ni, j) = vectorQ[j];
        }
      }

      // compute geometry
      double scs_error = 0.0;
      meSCS->determinant(1, &ws_coordinates(0, 0), &ws_scs_areav(0, 0), &scs_error);

      // start assembly
      for ( int ip = 0; ip < numScsIp; ++ip ) {

        // left and right nodes for this ip
        const int il = lrscv[2*ip];
        const int ir = lrscv[2*ip+1];

        stk::mesh::Entity nodeL = node_rels[il];
        stk::mesh::Entity nodeR = node_rels[ir];

        // pointer to fields to assemble
        double *gradQL = stk::mesh::field_data(*dqdx_, nodeL);
        double *gradQR = stk::mesh::field_data(*dqdx_, nodeR);

        // interpolate to scs point; operate on saved off ws_field
        for (int j=0; j < nDim; ++j )
          qIp[j] = 0.0;

        for ( int ic = 0; ic < nodesPerElement; ++ic ) {
          const double r = ws_shape_function(ip, ic);
          for ( int j = 0; j < nDim; ++j ) {
            qIp[j] += r*ws_vectorQ(ic, j);
          }
        }

        // left and right volume
        double inv_volL = 1.0/ws_dualVolume[il];
        double inv_volR = 1.0/ws_dualVolume[ir];

        // assemble to il/ir
        for ( int i = 0; i < nDim; ++i ) {
          const int row_gradQ = i*nDim;
          const double qip = qIp[i];
          for ( int j = 0; j < nDim; ++j ) {
            double fac = qip*ws_scs_areav(ip, j);
            Kokkos::atomic_add(&gradQL[row_gradQ+j], fac*inv_volL);
            Kokkos::atomic_sub(&gradQR[row_gradQ+j], fac*inv_volR);
          }
        }
      }
    });
  });
}

} // namespace nalu
} // namespace Sierra
