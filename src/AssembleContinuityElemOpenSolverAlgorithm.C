/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


// nalu
#include <AssembleContinuityElemOpenSolverAlgorithm.h>
#include <EquationSystem.h>
#include <FieldTypeDef.h>
#include <LinearSystem.h>
#include <Realm.h>
#include <master_element/MasterElement.h>

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
// AssembleContinuityElemOpenSolverAlgorithm - lhs for continuity open bc
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
AssembleContinuityElemOpenSolverAlgorithm::AssembleContinuityElemOpenSolverAlgorithm(
  Realm &realm,
  stk::mesh::Part *part,
  EquationSystem *eqSystem)
  : SolverAlgorithm(realm, part, eqSystem),
    meshMotion_(realm_.does_mesh_move()),
    velocityRTM_(NULL),
    Gpdx_(NULL),
    coordinates_(NULL),
    pressure_(NULL),
    density_(NULL),
    exposedAreaVec_(NULL),
    pressureBc_(NULL),
    shiftMdot_(realm_.get_cvfem_shifted_mdot()),
    shiftPoisson_(realm_.get_cvfem_shifted_poisson()),
    reducedSensitivities_(realm_.get_cvfem_reduced_sens_poisson())
{
  // save off fields
  stk::mesh::MetaData & meta_data = realm_.meta_data();
  if ( meshMotion_ )
    velocityRTM_ = meta_data.get_field<VectorFieldType>(stk::topology::NODE_RANK, "velocity_rtm");
  else
    velocityRTM_ = meta_data.get_field<VectorFieldType>(stk::topology::NODE_RANK, "velocity");
  Gpdx_ = meta_data.get_field<VectorFieldType>(stk::topology::NODE_RANK, "dpdx");
  coordinates_ = meta_data.get_field<VectorFieldType>(stk::topology::NODE_RANK, realm_.get_coordinates_name());
  pressure_ = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "pressure");
  density_ = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "density");
  exposedAreaVec_ = meta_data.get_field<GenericFieldType>(meta_data.side_rank(), "exposed_area_vector");
  pressureBc_ = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "pressure_bc");

  // Implementation details: code is designed to manage the following
  // When shiftPoisson_ is TRUE, reducedSensitivities_ is enforced to be TRUE
  // However, shiftPoisson_ can be FALSE while reducedSensitivities_ is TRUE
}

//--------------------------------------------------------------------------
//-------- initialize_connectivity -----------------------------------------
//--------------------------------------------------------------------------
void
AssembleContinuityElemOpenSolverAlgorithm::initialize_connectivity()
{
  eqSystem_->linsys_->buildFaceElemToNodeGraph(partVec_);
}

//--------------------------------------------------------------------------
//-------- execute ---------------------------------------------------------
//--------------------------------------------------------------------------
void
AssembleContinuityElemOpenSolverAlgorithm::execute()
{

  stk::mesh::BulkData & bulk_data = realm_.bulk_data();
  stk::mesh::MetaData & meta_data = realm_.meta_data();

  const int nDim = meta_data.spatial_dimension();
  
  // extract noc
  const std::string dofName = "pressure";
  const double includeNOC 
    = (realm_.get_noc_usage(dofName) == true) ? 1.0 : 0.0;


  const int maxElementsPerBucket = 512;
  const int maxNodesPerElement = 8;
  const int maxNodesPerFace = 4;
  const int maxNumScsIp = 16;
  const int maxDim = 3;
  const int maxlhsSize = maxNodesPerElement*maxNodesPerElement;
  const int maxrhsSize = maxNodesPerElement;

  const int bytes_per_team = SharedMemView<double *>::shmem_size(2*maxNumScsIp * maxNodesPerElement + maxNumScsIp * maxNodesPerFace);
  // TODO: This may substantially overestimate the scratch space needed depending on what
  // element types are actually present. We should investigate whether the cost of this matters
  // and if so consider the Aria approach where a separate algorithm is created per topology.
  const int bytes_per_thread =
      SharedMemView<double *>::shmem_size(maxNodesPerElement*nDim) + //ws_coordinates
      SharedMemView<double *>::shmem_size(maxNodesPerElement) + //ws_pressure
      SharedMemView<double *>::shmem_size(maxNodesPerFace*nDim) + //ws_vrtm
      SharedMemView<double *>::shmem_size(maxNodesPerFace*nDim) + //ws_Gpdx
      SharedMemView<double *>::shmem_size(maxNodesPerFace) + //ws_density
      SharedMemView<double *>::shmem_size(maxNodesPerFace) + //ws_bcPressure
      SharedMemView<double *>::shmem_size(maxDim*maxNumScsIp*maxNodesPerElement) +
      SharedMemView<double *>::shmem_size(maxDim*maxNumScsIp*maxNodesPerElement) +
      SharedMemView<double *>::shmem_size(maxNumScsIp) +
      SharedMemView<double *>::shmem_size(maxlhsSize) + //lhs
      SharedMemView<double *>::shmem_size(maxrhsSize) + //rhs
      SharedMemView<double *>::shmem_size(nDim) + //uBip
      SharedMemView<double *>::shmem_size(nDim) + //rho_uBip
      SharedMemView<double *>::shmem_size(nDim) + //GpdxBip
      SharedMemView<double *>::shmem_size(nDim) + //coordBip
      SharedMemView<double *>::shmem_size(nDim) + //coordScs
      SharedMemView<int *>::shmem_size(maxNodesPerFace); // face_node_ordina_vec
      SharedMemView<stk::mesh::Entity *>::shmem_size(maxNodesPerElement) + //entities
      SharedMemView<int *>::shmem_size(maxrhsSize); // For TpetraLinearSystem::sumInto vector of localIds


  // time step
  const double dt = realm_.get_time_step();
  const double gamma1 = realm_.get_gamma1();
  const double projTimeScale = dt/gamma1;

  // deal with interpolation procedure
  const double interpTogether = realm_.get_mdot_interp();
  const double om_interpTogether = 1.0-interpTogether;

  // deal with state
  ScalarFieldType &densityNp1 = density_->field_of_state(stk::mesh::StateNP1);

  // define vector of parent topos; should always be UNITY in size

  // define some common selectors
  stk::mesh::Selector s_locally_owned_union = meta_data.locally_owned_part()
    &stk::mesh::selectUnion(partVec_);

  stk::mesh::BucketVector const& face_buckets =
    realm_.get_buckets( meta_data.side_rank(), s_locally_owned_union );
  //KOKKOS noparallel BucketLoop: shared scratch arrays, calls sumInto from Tpetra, throws
  auto team_exec = get_team_policy(face_buckets.size(), bytes_per_team, bytes_per_thread);

  Kokkos::parallel_for("Nalu::AssembleContinuityElemOpenSolver",
      team_exec, [&] (const DeviceTeam & team)
  {
    const int ib = team.league_rank();
    const stk::mesh::Bucket & b = *face_buckets[ib];
    const stk::mesh::Bucket::size_type length = b.size();

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

    // resize some things; matrix related
    const int lhsSize = nodesPerElement*nodesPerElement;
    const int rhsSize = nodesPerElement;

    SharedMemView<double*> ws_shape_function(team.team_shmem(), numScsIp*nodesPerElement);
    SharedMemView<double*> ws_shape_function_lhs(team.team_shmem(), numScsIp*nodesPerElement);
    SharedMemView<double*> ws_face_shape_function(team.team_shmem(), numScsBip*nodesPerFace);

    SharedMemView<stk::mesh::Entity*> connected_nodes;
    SharedMemView<int*> face_node_ordinal_vec;
    SharedMemView<double*> lhs;
    SharedMemView<double*> rhs;
    SharedMemView<int*> localIdsScratch;
    SharedMemView<double*> ws_vrtm;
    SharedMemView<double*> ws_Gpdx;
    SharedMemView<double*> ws_coordinates;
    SharedMemView<double*> ws_pressure;
    SharedMemView<double*> ws_density;
    SharedMemView<double*> ws_bcPressure;
    SharedMemView<double*> uBip;
    SharedMemView<double*> rho_uBip;
    SharedMemView<double*> GpdxBip;
    SharedMemView<double*> coordBip;
    SharedMemView<double*> coordScs;
    {
      lhs = Kokkos::subview(
          SharedMemView<double**>(team.team_shmem(), team.team_size(), lhsSize),
          team.team_rank(), Kokkos::ALL());
      rhs = Kokkos::subview(
          SharedMemView<double**>(team.team_shmem(), team.team_size(), rhsSize),
          team.team_rank(), Kokkos::ALL());
      connected_nodes = Kokkos::subview(
          SharedMemView<stk::mesh::Entity**> (team.team_shmem(), team.team_size(), nodesPerElement),
          team.team_rank(), Kokkos::ALL());
      ws_vrtm = Kokkos::subview(
          SharedMemView<double**>(team.team_shmem(), team.team_size(), nodesPerFace*nDim),
          team.team_rank(), Kokkos::ALL());
      ws_Gpdx = Kokkos::subview(
          SharedMemView<double**>(team.team_shmem(), team.team_size(), nodesPerFace*nDim),
          team.team_rank(), Kokkos::ALL());
      ws_coordinates = Kokkos::subview(
          SharedMemView<double**>(team.team_shmem(), team.team_size(), nodesPerElement*nDim),
          team.team_rank(), Kokkos::ALL());
      ws_bcPressure = Kokkos::subview(
          SharedMemView<double**>(team.team_shmem(), team.team_size(), nodesPerFace),
          team.team_rank(), Kokkos::ALL());
      ws_pressure = Kokkos::subview(
          SharedMemView<double**>(team.team_shmem(), team.team_size(), nodesPerElement),
          team.team_rank(), Kokkos::ALL());
      ws_density = Kokkos::subview(
          SharedMemView<double**>(team.team_shmem(), team.team_size(), nodesPerFace),
          team.team_rank(), Kokkos::ALL());
      uBip = Kokkos::subview(
          SharedMemView<double**>(team.team_shmem(), team.team_size(), nDim),
          team.team_rank(), Kokkos::ALL());
      rho_uBip = Kokkos::subview(
          SharedMemView<double**>(team.team_shmem(), team.team_size(), nDim),
          team.team_rank(), Kokkos::ALL());
      GpdxBip = Kokkos::subview(
          SharedMemView<double**>(team.team_shmem(), team.team_size(), nDim),
          team.team_rank(), Kokkos::ALL());
      coordBip = Kokkos::subview(
          SharedMemView<double**>(team.team_shmem(), team.team_size(), nDim),
          team.team_rank(), Kokkos::ALL());
      coordScs = Kokkos::subview(
          SharedMemView<double**>(team.team_shmem(), team.team_size(), nDim),
          team.team_rank(), Kokkos::ALL());
      localIdsScratch = Kokkos::subview(
          SharedMemView<int**> (team.team_shmem(), team.team_size(), rhsSize),
          team.team_rank(), Kokkos::ALL());
      face_node_ordinal_vec = Kokkos::subview(
          SharedMemView<int**> (team.team_shmem(), team.team_size(), nodesPerFace),
          team.team_rank(), Kokkos::ALL());
    }

    // shape functions; interior
    Kokkos::single(Kokkos::PerTeam(team), [&]() {
      if ( shiftPoisson_ )
        meSCS->shifted_shape_fcn(&ws_shape_function[0]);
      else
        meSCS->shape_fcn(&ws_shape_function[0]);

      double *p_shape_function_lhs = shiftPoisson_ ? &ws_shape_function[0] : reducedSensitivities_ ? &ws_shape_function_lhs[0] : &ws_shape_function[0];
      if ( !shiftPoisson_ && reducedSensitivities_ )
        meSCS->shifted_shape_fcn(&p_shape_function_lhs[0]);

      // shape functions; boundary
      if ( shiftMdot_ )
        meFC->shifted_shape_fcn(&ws_face_shape_function[0]);
      else
        meFC->shape_fcn(&ws_face_shape_function[0]);
    });
    team.team_barrier();

    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, length), [&] (const size_t k)
    {
      // zero lhs/rhs
      for ( int p = 0; p < lhsSize; ++p )
        lhs[p] = 0.0;
      for ( int p = 0; p < rhsSize; ++p )
        rhs[p] = 0.0;

      // get face
      stk::mesh::Entity face = b[k];

      //======================================
      // gather nodal data off of face
      //======================================
      stk::mesh::Entity const * face_node_rels = bulk_data.begin_nodes(face);
      int num_face_nodes = bulk_data.num_nodes(face);
      // sanity check on num nodes
      for ( int ni = 0; ni < num_face_nodes; ++ni ) {
        stk::mesh::Entity node = face_node_rels[ni];

        // gather scalars
        ws_density[ni] = *stk::mesh::field_data(densityNp1, node);
        ws_bcPressure[ni] = *stk::mesh::field_data(*pressureBc_, node);

        // gather vectors
        const double * vrtm = stk::mesh::field_data(*velocityRTM_, node);
        const double * Gjp = stk::mesh::field_data(*Gpdx_, node);
        const int offSet = ni*nDim;
        for ( int j=0; j < nDim; ++j ) {
          ws_vrtm[offSet+j] = vrtm[j];
          ws_Gpdx[offSet+j] = Gjp[j];
        }
      }

      // pointer to face data
      const double * areaVec = stk::mesh::field_data(*exposedAreaVec_, face);

      // extract the connected element to this exposed face; should be single in size!
      const stk::mesh::Entity* face_elem_rels = bulk_data.begin_elements(face);

      // get element; its face ordinal number and populate face_node_ordinal_vec
      stk::mesh::Entity element = face_elem_rels[0];
      const stk::mesh::ConnectivityOrdinal* face_elem_ords = bulk_data.begin_element_ordinals(face);
      const int face_ordinal = face_elem_ords[0];
      theElemTopo.side_node_ordinals(face_ordinal, &face_node_ordinal_vec(0));


      // mapping from ip to nodes for this ordinal
      const int *ipNodeMap = meSCS->ipNodeMap(face_ordinal);

      //======================================
      // gather nodal data off of element
      //======================================
      stk::mesh::Entity const * elem_node_rels = bulk_data.begin_nodes(element);
      int num_nodes = bulk_data.num_nodes(element);
      // sanity check on num nodes
      for ( int ni = 0; ni < num_nodes; ++ni ) {
        stk::mesh::Entity node = elem_node_rels[ni];

        // set connected nodes
        connected_nodes[ni] = node;

        // gather scalars
        ws_pressure[ni] = *stk::mesh::field_data(*pressure_, node);

        // gather vectors
        const double * coords = stk::mesh::field_data(*coordinates_, node);
        const int offSet = ni*nDim;
        for ( int j=0; j < nDim; ++j ) {
          ws_coordinates[offSet+j] = coords[j];
        }
      }

      // loop over boundary ips
      for ( int ip = 0; ip < numScsBip; ++ip ) {

        const int nearestNode = ipNodeMap[ip];
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
          const double r = ws_face_shape_function[offSetSF_face+ic];
          const double rhoIC = ws_density[ic];
          rhoBip += r*rhoIC;
          pBip += r*ws_bcPressure[ic];
          const int offSetFN = ic*nDim;
          const int offSetEN = fn*nDim;
          for ( int j = 0; j < nDim; ++j ) {
            uBip[j] += r*ws_vrtm[offSetFN+j];
            rho_uBip[j] += r*rhoIC*ws_vrtm[offSetFN+j];
            GpdxBip[j] += r*ws_Gpdx[offSetFN+j];
            coordBip[j] += r*ws_coordinates[offSetEN+j];
          }
        }

        // data at interior opposing face
        double pScs = 0.0;
        const int offSetSF_elem = opposingScsIp*nodesPerElement;
        for ( int ic = 0; ic < nodesPerElement; ++ic ) {
          const double r = ws_shape_function[offSetSF_elem+ic];
          pScs += r*ws_pressure[ic];
          const int offSet = ic*nDim;
          for ( int j = 0; j < nDim; ++j ) {
            coordScs[j] += r*ws_coordinates[offSet+j];
          }
        }

        // form axdx, asq and mdot (without dp/dn or noc)
        double asq = 0.0;
        double axdx = 0.0;
        double mdot = 0.0;
        for ( int j = 0; j < nDim; ++j ) {
          const double dxj = coordBip[j] - coordScs[j];
          const double axj = areaVec[ip*nDim+j];
          asq += axj*axj;
          axdx += axj*dxj;
          mdot += (interpTogether*rho_uBip[j] + om_interpTogether*rhoBip*uBip[j]
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

        // lhs for pressure system
        int rowR = nearestNode*nodesPerElement;

        double *p_shape_function_lhs = shiftPoisson_ ? &ws_shape_function[0] : reducedSensitivities_ ? &ws_shape_function_lhs[0] : &ws_shape_function[0];
        for ( int ic = 0; ic < nodesPerElement; ++ic ) {
          const double r = p_shape_function_lhs[offSetSF_elem+ic];
          lhs[rowR+ic] += r*asq*inv_axdx;
        }

        // final mdot
        mdot += -projTimeScale*((pBip-pScs)*asq*inv_axdx + noc*includeNOC);

        // residual
        rhs[nearestNode] -= mdot/projTimeScale;
      }

      apply_coeff(connected_nodes, rhs, lhs, localIdsScratch, __FILE__);

    });
  });
}

} // namespace nalu
} // namespace Sierra
