/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


// nalu
#include <AssembleCourantReynoldsElemAlgorithm.h>
#include <Algorithm.h>

#include <FieldTypeDef.h>
#include <Realm.h>
#include <NaluEnv.h>
#include <TimeIntegrator.h>
#include <master_element/MasterElement.h>

// stk_mesh/base/fem
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Field.hpp>
#include <stk_mesh/base/GetBuckets.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/Part.hpp>

// stk_util
#include <stk_util/parallel/ParallelReduce.hpp>

#include <KokkosInterface.h>

namespace sierra{
namespace nalu{

//==========================================================================
// Class Definition
//==========================================================================
// AssembleCourantReynoldsElemAlgorithm - Courant number calc for both edge
//                                        and elem
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
AssembleCourantReynoldsElemAlgorithm::AssembleCourantReynoldsElemAlgorithm(
  Realm &realm,
  stk::mesh::Part *part)
  : Algorithm(realm, part),
    meshMotion_(realm_.does_mesh_move()),
    velocityRTM_(NULL),
    coordinates_(NULL),
    density_(NULL),
    viscosity_(NULL),
    elemReynolds_(NULL),
    elemCourant_(NULL)
{
  // save off data
  stk::mesh::MetaData & meta_data = realm_.meta_data();
  if ( meshMotion_ )
    velocityRTM_ = meta_data.get_field<VectorFieldType>(stk::topology::NODE_RANK, "velocity_rtm");
  else
    velocityRTM_ = meta_data.get_field<VectorFieldType>(stk::topology::NODE_RANK, "velocity");
  coordinates_ = meta_data.get_field<VectorFieldType>(stk::topology::NODE_RANK, realm_.get_coordinates_name());
  density_ = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "density");
  const std::string viscName = (realm.is_turbulent())
     ? "effective_viscosity_u" : "viscosity";
  viscosity_ = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, viscName);

  // provide for elemental fields
  elemReynolds_ = meta_data.get_field<GenericFieldType>(stk::topology::ELEMENT_RANK, "element_reynolds");
  elemCourant_ = meta_data.get_field<GenericFieldType>(stk::topology::ELEMENT_RANK, "element_courant");
}

//--------------------------------------------------------------------------
//-------- execute ---------------------------------------------------------
//--------------------------------------------------------------------------
void
AssembleCourantReynoldsElemAlgorithm::execute()
{

  stk::mesh::BulkData & bulk_data = realm_.bulk_data();
  stk::mesh::MetaData & meta_data = realm_.meta_data();

  const int nDim = meta_data.spatial_dimension();

  const double dt = realm_.timeIntegrator_->get_time_step();
  const double small = 1.0e-16;

  // deal with state
  ScalarFieldType &densityNp1 = density_->field_of_state(stk::mesh::StateNP1);

  // set courant/reynolds number to something small
  double maxCR[2] = {-1.0, -1.0};

  // define some common selectors
  stk::mesh::Selector s_locally_owned_union = meta_data.locally_owned_part()
    & stk::mesh::selectUnion(partVec_) 
    & !(realm_.get_inactive_selector());

  stk::mesh::BucketVector const& elem_buckets =
    realm_.get_buckets( stk::topology::ELEMENT_RANK, s_locally_owned_union );

  const int maxNodesPerElement = 8;
  const int bytes_per_thread =
      SharedMemView<double **>::shmem_size(maxNodesPerElement, nDim) +
      SharedMemView<double **>::shmem_size(maxNodesPerElement, nDim) +
      SharedMemView<double *>::shmem_size(maxNodesPerElement) +
      SharedMemView<double *>::shmem_size(maxNodesPerElement);

  auto team_exec = get_team_policy(elem_buckets.size(), 0, bytes_per_thread);

  Kokkos::parallel_for("AssembleCourantReynoldsElemAlgorithm::execute",
    team_exec, [&](const DeviceTeam & team) {
    const stk::mesh::Bucket & b = *elem_buckets[team.league_rank()];
    const stk::mesh::Bucket::size_type length   = b.size();

    // extract master element
    MasterElement *meSCS = realm_.get_surface_master_element(b.topology());

    // extract master element specifics
    const int nodesPerElement = meSCS->nodesPerElement_;
    const int numScsIp = meSCS->numIntPoints_;
    const int *lrscv = meSCS->adjacentNodes();

    // algorithm related
    const int scratch_level = 2;
    SharedMemView<double **> vrtm(team.thread_scratch(scratch_level), nodesPerElement, nDim);
    SharedMemView<double **> coordinates(team.thread_scratch(scratch_level), nodesPerElement, nDim);
    SharedMemView<double *> density(team.thread_scratch(scratch_level), nodesPerElement);
    SharedMemView<double *> viscosity(team.thread_scratch(scratch_level), nodesPerElement);

    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, length), [&](const size_t k) {
      // get elem
      stk::mesh::Entity elem = b[k];
      
      // pointers to elem data
      double * elemReynolds = stk::mesh::field_data(*elemReynolds_, b, k);
      double * elemCourant = stk::mesh::field_data(*elemCourant_, b, k);
      
      //===============================================
      // gather nodal data; this is how we do it now..
      //===============================================
      stk::mesh::Entity const * node_rels = bulk_data.begin_nodes(elem);
      int num_nodes = bulk_data.num_nodes(elem);

      // sanity check on num nodes
      //ThrowAssert( num_nodes == nodesPerElement );

      for ( int ni = 0; ni < num_nodes; ++ni ) {
        stk::mesh::Entity node = node_rels[ni];

        // pointers to real data
        double * coords = stk::mesh::field_data(*coordinates_, node );
        double * vrtm_field = stk::mesh::field_data(*velocityRTM_, node );

        // gather scalars
        density[ni]   = *stk::mesh::field_data(densityNp1, node);
        viscosity[ni] = *stk::mesh::field_data(*viscosity_, node);

        // gather vectors
        for ( int j = 0; j < nDim; ++j ) {
          coordinates(ni, j) = coords[j];
          vrtm(ni, j) = vrtm_field[j];
        }
      }

      // compute cfl and Re along each edge; set ip max to negative
      double eReynolds = -1.0;
      double eCourant = -1.0;
      for ( int ip = 0; ip < numScsIp; ++ip ) {

        // left and right nodes for this ip
        const int il = lrscv[2*ip];
        const int ir = lrscv[2*ip+1];

        double udotx = 0.0;
        double dxSq = 0.0;
        for ( int j = 0; j < nDim; ++j ) {
          double ujIp = 0.5*(vrtm(ir, j)+vrtm(il, j));
          double dxj = coordinates(ir, j) - coordinates(il, j);
          udotx += dxj*ujIp;
          dxSq += dxj*dxj;
        }

        udotx = std::abs(udotx);
        const double ipCourant = std::abs(udotx*dt/dxSq);
        maxCR[0] = std::max(maxCR[0], ipCourant);

        const double diffIp = 0.5*( viscosity[il]/density[il] + viscosity[ir]/density[ir] );
        const double ipReynolds = udotx/(diffIp+small);
        maxCR[1] = std::max(maxCR[1], ipReynolds);
        
        // determine local max ip value
        eReynolds = std::max(eReynolds, ipReynolds);
        eCourant = std::max(eCourant, ipCourant);
      }
      
      // scatter
      elemReynolds[0] = eReynolds;
      elemCourant[0] = eCourant;
    });
  });

  // parallel max
  double g_maxCR[2]  = {};
  stk::ParallelMachine comm = NaluEnv::self().parallel_comm();
  stk::all_reduce_max(comm, maxCR, g_maxCR, 2);

  // sent to realm
  realm_.maxCourant_ = g_maxCR[0];
  realm_.maxReynolds_ = g_maxCR[1];

}

} // namespace nalu
} // namespace Sierra
