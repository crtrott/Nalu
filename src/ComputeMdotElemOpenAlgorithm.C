/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


// nalu
#include <ComputeMdotElemOpenAlgorithm.h>
#include <Algorithm.h>

#include <FieldTypeDef.h>
#include <Realm.h>
#include <master_element/MasterElement.h>

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
// ComputeMdotElemOpenAlgorithm - mdot continuity open bc
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
ComputeMdotElemOpenAlgorithm::ComputeMdotElemOpenAlgorithm(
  Realm &realm,
  stk::mesh::Part *part)
  : Algorithm(realm, part),
    velocity_(NULL),
    Gpdx_(NULL),
    coordinates_(NULL),
    pressure_(NULL),
    density_(NULL),
    exposedAreaVec_(NULL),
    pressureBc_(NULL),
    shiftMdot_(realm_.get_cvfem_shifted_mdot()),
    shiftPoisson_(realm_.get_cvfem_shifted_poisson())
{
  // save off fields
  stk::mesh::MetaData & meta_data = realm_.meta_data();
  velocity_ = meta_data.get_field<VectorFieldType>(stk::topology::NODE_RANK, "velocity");
  Gpdx_ = meta_data.get_field<VectorFieldType>(stk::topology::NODE_RANK, "dpdx");
  coordinates_ = meta_data.get_field<VectorFieldType>(stk::topology::NODE_RANK, realm_.get_coordinates_name());
  pressure_ = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "pressure");
  density_ = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "density");
  exposedAreaVec_ = meta_data.get_field<GenericFieldType>(meta_data.side_rank(), "exposed_area_vector");
  openMassFlowRate_ = meta_data.get_field<GenericFieldType>(meta_data.side_rank(), "open_mass_flow_rate");
  pressureBc_ = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "pressure_bc");
}

//--------------------------------------------------------------------------
//-------- execute ---------------------------------------------------------
//--------------------------------------------------------------------------
void
ComputeMdotElemOpenAlgorithm::execute()
{
  stk::mesh::BulkData & bulk_data = realm_.bulk_data();
  stk::mesh::MetaData & meta_data = realm_.meta_data();

  const int nDim = meta_data.spatial_dimension();
  
  // extract noc
  const std::string dofName = "pressure";
  const double includeNOC 
    = (realm_.get_noc_usage(dofName) == true) ? 1.0 : 0.0;

  // time step
  const double dt = realm_.get_time_step();
  const double gamma1 = realm_.get_gamma1();
  const double projTimeScale = dt/gamma1;

  // deal with interpolation procedure
  const double interpTogether = realm_.get_mdot_interp();
  const double om_interpTogether = 1.0-interpTogether;

  // deal with state
  VectorFieldType &velocityNp1 = velocity_->field_of_state(stk::mesh::StateNP1);
  ScalarFieldType &densityNp1 = density_->field_of_state(stk::mesh::StateNP1);

  // define vector of parent topos; should always be UNITY in size
  std::vector<stk::topology> parentTopo;

  // define some common selectors
  stk::mesh::Selector s_locally_owned_union = meta_data.locally_owned_part()
    &stk::mesh::selectUnion(partVec_);

  stk::mesh::BucketVector const& face_buckets =
    realm_.get_buckets( meta_data.side_rank(), s_locally_owned_union );

  const int maxNodesPerElement = 8;
  const int maxNumScsIp = 16;
  const int maxNodesPerFace = 4;
  const int maxNumScsBip = 8;

  const int bytes_per_thread = SharedMemView<int *>::shmem_size(maxNodesPerFace) +
      SharedMemView<double *>::shmem_size(nDim) +
      SharedMemView<double *>::shmem_size(nDim) +
      SharedMemView<double *>::shmem_size(nDim) +
      SharedMemView<double *>::shmem_size(nDim) +
      SharedMemView<double *>::shmem_size(nDim) +
      SharedMemView<double **>::shmem_size(maxNodesPerElement, nDim) +
      SharedMemView<double *>::shmem_size(maxNodesPerElement) +
      SharedMemView<double **>::shmem_size(maxNodesPerFace, nDim) +
      SharedMemView<double **>::shmem_size(maxNodesPerFace, nDim) +
      SharedMemView<double *>::shmem_size(maxNodesPerFace) +
      SharedMemView<double *>::shmem_size(maxNodesPerFace);

  const int bytes_per_team =
      SharedMemView<double **>::shmem_size(maxNumScsIp, maxNodesPerElement) +
      SharedMemView<double **>::shmem_size(maxNumScsBip, maxNodesPerFace);

  auto team_exec = get_team_policy(face_buckets.size(), bytes_per_team, bytes_per_thread);

  Kokkos::parallel_for("ComputeMdotElemOpenAlgorithm::execute",
      team_exec, [&](const DeviceTeam & team) {
    stk::mesh::Bucket & b = *face_buckets[team.league_rank()];
    const auto length = b.size();

    // extract connected element topology
    const auto first_elem = bulk_data.begin_elements(b[0])[0];
    stk::topology theElemTopo = bulk_data.bucket(first_elem).topology();

    // volume master element
    MasterElement *meSCS = realm_.get_surface_master_element(theElemTopo);
    const int nodesPerElement = meSCS->nodesPerElement_;
    const int numScsIp = meSCS->numIntPoints_;

    // face master element
    MasterElement *meFC = realm_.get_surface_master_element(b.topology());
    const int nodesPerFace = b.topology().num_nodes();
    const int numScsBip = meFC->numIntPoints_;

    // algorithm related; element
    const int scratch_level = 2;
    SharedMemView<int *> face_node_ordinal_vec(team.thread_scratch(scratch_level), nodesPerFace);
    SharedMemView<double *> uBip(team.thread_scratch(scratch_level), nDim);
    SharedMemView<double *> rho_uBip(team.thread_scratch(scratch_level), nDim);
    SharedMemView<double *> GpdxBip(team.thread_scratch(scratch_level), nDim);
    SharedMemView<double *> coordBip(team.thread_scratch(scratch_level), nDim);
    SharedMemView<double *> coordScs(team.thread_scratch(scratch_level), nDim);
    SharedMemView<double **> ws_coordinates(team.thread_scratch(scratch_level), nodesPerElement, nDim);
    SharedMemView<double *> ws_pressure(team.thread_scratch(scratch_level), nodesPerElement);
    SharedMemView<double **> ws_velocityNp1(team.thread_scratch(scratch_level), nodesPerFace, nDim);
    SharedMemView<double **> ws_Gpdx(team.thread_scratch(scratch_level), nodesPerFace, nDim);
    SharedMemView<double *> ws_density(team.thread_scratch(scratch_level), nodesPerFace);
    SharedMemView<double *> ws_bcPressure(team.thread_scratch(scratch_level), nodesPerFace);

    SharedMemView<double **> ws_shape_function(team.team_scratch(scratch_level), numScsIp, nodesPerElement);
    SharedMemView<double **> ws_face_shape_function(team.team_scratch(scratch_level), numScsBip, nodesPerFace);

    Kokkos::single(Kokkos::PerTeam(team), [&]() {
      // shape functions; interior
      if ( shiftPoisson_ )
        meSCS->shifted_shape_fcn(&ws_shape_function(0, 0));
      else
        meSCS->shape_fcn(&ws_shape_function(0, 0));

      // shape functions; boundary
      if ( shiftMdot_ )
        meFC->shifted_shape_fcn(&ws_face_shape_function(0, 0));
      else
        meFC->shape_fcn(&ws_face_shape_function(0, 0));
    });
    team.team_barrier();
    
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, length), [&](const size_t k) {
      // get face
      stk::mesh::Entity face = b[k];

      //======================================
      // gather nodal data off of face
      //======================================
      stk::mesh::Entity const * face_node_rels = bulk_data.begin_nodes(face);
      int num_face_nodes = bulk_data.num_nodes(face);
      // sanity check on num nodes
      ThrowAssert( num_face_nodes == nodesPerFace );
      for ( int ni = 0; ni < num_face_nodes; ++ni ) {
        stk::mesh::Entity node = face_node_rels[ni];

        // gather scalars
        ws_density[ni]    = *stk::mesh::field_data(densityNp1, node);
        ws_bcPressure[ni] = *stk::mesh::field_data(*pressureBc_, node);

        // gather vectors
        double * uNp1 = stk::mesh::field_data(velocityNp1, node);
        double * Gjp = stk::mesh::field_data(*Gpdx_, node);
        const int offSet = ni*nDim;
        for ( int j=0; j < nDim; ++j ) {
          ws_velocityNp1(ni, j) = uNp1[j];
          ws_Gpdx(ni, j) = Gjp[j];
        }
      }

      // pointer to face data
      const double * areaVec = stk::mesh::field_data(*exposedAreaVec_, face);
      double * mdot = stk::mesh::field_data(*openMassFlowRate_, face);

      // extract the connected element to this exposed face; should be single in size!
      const stk::mesh::Entity* face_elem_rels = bulk_data.begin_elements(face);
      //ThrowAssert( bulk_data.num_elements(face) == 1 );

      // get element; its face ordinal number and populate face_node_ordinal_vec
      stk::mesh::Entity element = face_elem_rels[0];
      const int face_ordinal = bulk_data.begin_element_ordinals(face)[0];
      theElemTopo.side_node_ordinals(face_ordinal, &face_node_ordinal_vec(0));

      //======================================
      // gather nodal data off of element
      //======================================
      stk::mesh::Entity const * elem_node_rels = bulk_data.begin_nodes(element);
      int num_nodes = bulk_data.num_nodes(element);
      // sanity check on num nodes
      ThrowAssert( num_nodes == nodesPerElement );
      for ( int ni = 0; ni < num_nodes; ++ni ) {
        stk::mesh::Entity node = elem_node_rels[ni];

        // gather scalars
        ws_pressure[ni] = *stk::mesh::field_data(*pressure_, node);

        // gather vectors
        double * coords = stk::mesh::field_data(*coordinates_, node);
        for ( int j=0; j < nDim; ++j ) {
          ws_coordinates(ni, j) = coords[j];
        }
      }

      // loop over boundary ips
      for ( int ip = 0; ip < numScsBip; ++ip ) {

        const int opposingScsIp = meSCS->opposingFace(face_ordinal,ip);

        // zero out vector quantities
        for ( int j = 0; j < nDim; ++j ) {
          uBip[j] = 0.0;
          rho_uBip[j] = 0.0;
          GpdxBip[j] = 0.0;
          coordBip[j] = 0.0;
          coordScs[j] = 0.0;
        }
        double rhoBip = 0.0;

        // interpolate to bip
        double pBip = 0.0;
        const int offSetSF_face = ip*nodesPerFace;
        for ( int ic = 0; ic < nodesPerFace; ++ic ) {
          const int fn = face_node_ordinal_vec[ic];
          const double r = ws_face_shape_function(ip, ic);
          const double rhoIC = ws_density[ic];
          rhoBip += r*rhoIC;
          pBip += r*ws_bcPressure[ic];
          for ( int j = 0; j < nDim; ++j ) {
            uBip[j] += r*ws_velocityNp1(ic, j);
            rho_uBip[j] += r*rhoIC*ws_velocityNp1(ic, j);
            GpdxBip[j] += r*ws_Gpdx(ic, j);
            coordBip[j] += r*ws_coordinates(fn, j);
          }
        }

        // data at interior opposing face
        double pScs = 0.0;
        for ( int ic = 0; ic < nodesPerElement; ++ic ) {
          const double r = ws_shape_function(opposingScsIp, ic);
          pScs += r*ws_pressure[ic];
          for ( int j = 0; j < nDim; ++j ) {
            coordScs[j] += r*ws_coordinates(ic, j);
          }
        }

        // form axdx, asq and mdot (without dp/dn or noc)
        double axdx = 0.0;
        double asq = 0.0;
        double tmdot = 0.0;
        for ( int j = 0; j < nDim; ++j ) {
          const double dxj = coordBip[j] - coordScs[j];
          const double axj = areaVec[ip*nDim+j];
          axdx += axj*dxj;
          asq += axj*axj;
          tmdot += (interpTogether*rho_uBip[j] + om_interpTogether*rhoBip*uBip[j]
                    + projTimeScale*GpdxBip[j])*axj;
        }
	
        const double inv_axdx = 1.0/axdx;

        // deal with noc
        double noc = 0.0;
        for ( int j = 0; j < nDim; ++j ) {
          const double dxj = coordBip[j] - coordScs[j];
          const double axj = areaVec[ip*nDim+j];
          const double kxj = axj - asq*inv_axdx*dxj; // NOC
          noc += kxj*GpdxBip[j];
        }

        // assign
        mdot[ip] = tmdot - projTimeScale*((pBip-pScs)*asq*inv_axdx + noc*includeNOC);

      }
    });
  });
}


//--------------------------------------------------------------------------
//-------- destructor ------------------------------------------------------
//--------------------------------------------------------------------------
ComputeMdotElemOpenAlgorithm::~ComputeMdotElemOpenAlgorithm()
{
  // does nothing
}



} // namespace nalu
} // namespace Sierra
