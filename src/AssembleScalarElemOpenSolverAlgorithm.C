/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


// nalu
#include <AssembleScalarElemOpenSolverAlgorithm.h>
#include <EquationSystem.h>
#include <FieldTypeDef.h>
#include <LinearSystem.h>
#include <PecletFunction.h>
#include <Realm.h>
#include <TimeIntegrator.h>
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
// AssembleScalarElemOpenSolverAlgorithm - lhs for scalar open bc
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
AssembleScalarElemOpenSolverAlgorithm::AssembleScalarElemOpenSolverAlgorithm(
  Realm &realm,
  stk::mesh::Part *part,
  EquationSystem *eqSystem,
  ScalarFieldType *scalarQ,
  ScalarFieldType *bcScalarQ,
  VectorFieldType *dqdx,
  ScalarFieldType *diffFluxCoeff)
  : SolverAlgorithm(realm, part, eqSystem),
    meshMotion_(realm.does_mesh_move()),
    scalarQ_(scalarQ),
    bcScalarQ_(bcScalarQ),
    dqdx_(dqdx),
    diffFluxCoeff_(diffFluxCoeff),
    velocityRTM_(NULL),
    coordinates_(NULL),
    density_(NULL),
    openMassFlowRate_(NULL),
    pecletFunction_(NULL)
{
  // save off fields
  stk::mesh::MetaData & meta_data = realm_.meta_data();
  if ( meshMotion_ )
    velocityRTM_ = meta_data.get_field<VectorFieldType>(stk::topology::NODE_RANK, "velocity_rtm");
  else
    velocityRTM_ = meta_data.get_field<VectorFieldType>(stk::topology::NODE_RANK, "velocity");
  coordinates_ = meta_data.get_field<VectorFieldType>(stk::topology::NODE_RANK, realm_.get_coordinates_name());
  density_ = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "density");
  openMassFlowRate_ = meta_data.get_field<GenericFieldType>(meta_data.side_rank(), "open_mass_flow_rate");

  // create the peclet blending function
  pecletFunction_ = eqSystem->create_peclet_function(scalarQ_->name());
}

//--------------------------------------------------------------------------
//-------- destructor ------------------------------------------------------
//--------------------------------------------------------------------------
AssembleScalarElemOpenSolverAlgorithm::~AssembleScalarElemOpenSolverAlgorithm()
{
  delete pecletFunction_;
}

//--------------------------------------------------------------------------
//-------- initialize_connectivity -----------------------------------------
//--------------------------------------------------------------------------
void
AssembleScalarElemOpenSolverAlgorithm::initialize_connectivity()
{
  eqSystem_->linsys_->buildFaceElemToNodeGraph(partVec_);
}

//--------------------------------------------------------------------------
//-------- execute ---------------------------------------------------------
//--------------------------------------------------------------------------
void
AssembleScalarElemOpenSolverAlgorithm::execute()
{

  stk::mesh::BulkData & bulk_data = realm_.bulk_data();
  stk::mesh::MetaData & meta_data = realm_.meta_data();

  const int nDim = meta_data.spatial_dimension();

  const double small = 1.0e-16;

  // extract user advection options (allow to potentially change over time)
  const std::string dofName = scalarQ_->name();
  const double alphaUpw = realm_.get_alpha_upw_factor(dofName);
  const double hoUpwind = realm_.get_upw_factor(dofName);

  // one minus flavor..
  const double om_alphaUpw = 1.0-alphaUpw;

  // deal with state
  ScalarFieldType &scalarQNp1 = scalarQ_->field_of_state(stk::mesh::StateNP1);
  ScalarFieldType &densityNp1 = density_->field_of_state(stk::mesh::StateNP1);

  // define some common selectors
  stk::mesh::Selector s_locally_owned_union = meta_data.locally_owned_part()
    &stk::mesh::selectUnion(partVec_);

  stk::mesh::BucketVector const& face_buckets =
    realm_.get_buckets( meta_data.side_rank(), s_locally_owned_union );

  const int maxNodesPerElement = 8;
  const int maxNodesPerFace = 4;
  const int maxNumScsBip = 8;

  const int maxlhsSize = maxNodesPerElement*nDim*maxNodesPerElement*nDim;
  const int maxrhsSize = maxNodesPerElement*nDim;

  const int bytes_per_thread =
      SharedMemView<double *>::shmem_size(maxlhsSize) +
      SharedMemView<double *>::shmem_size(maxrhsSize) +
      SharedMemView<stk::mesh::Entity *>::shmem_size(maxNodesPerElement) +
      SharedMemView<int *>::shmem_size(maxrhsSize) +
      SharedMemView<double **>::shmem_size(maxNodesPerFace, nDim) +
      SharedMemView<double *>::shmem_size(maxNodesPerFace) +
      SharedMemView<double *>::shmem_size(maxNodesPerFace) +
      SharedMemView<double *>::shmem_size(nDim) +
      SharedMemView<int *>::shmem_size(maxNodesPerFace);

  // shape functions
  const int bytes_per_team =
      SharedMemView<double **>::shmem_size(maxNumScsBip, maxNodesPerFace);

  auto team_exec = get_team_policy(face_buckets.size(), bytes_per_team, bytes_per_thread);

  Kokkos::parallel_for("Nalu::AssembleScalarElemOpenSolver",
      team_exec, [&](const DeviceTeam & team) {
    const stk::mesh::Bucket & b = *face_buckets[team.league_rank()];
    const auto length = b.size();

    // extract connected element topology
    const auto first_elem = bulk_data.begin_elements(b[0])[0];
    stk::topology theElemTopo = bulk_data.bucket(first_elem).topology();

    // volume master element
    MasterElement *meSCS = realm_.get_surface_master_element(theElemTopo);
    const int nodesPerElement = meSCS->nodesPerElement_;

    // face master element
    MasterElement *meFC = realm_.get_surface_master_element(b.topology());
    const int nodesPerFace = meFC->nodesPerElement_;
    const int numScsBip = meFC->numIntPoints_;

    // resize some things; matrix related
    const int lhsSize = nodesPerElement*nodesPerElement;
    const int rhsSize = nodesPerElement;
    const int scratch_level = 2;
    SharedMemView<double *> lhs(team.thread_scratch(scratch_level), lhsSize);
    SharedMemView<double *> rhs(team.thread_scratch(scratch_level), rhsSize);
    SharedMemView<stk::mesh::Entity *> connected_nodes(team.thread_scratch(scratch_level), nodesPerElement);
    SharedMemView<int *> localIdsScratch(team.thread_scratch(scratch_level), rhsSize);

    // algorithm related; element
    SharedMemView<double **> ws_face_coordinates(team.thread_scratch(scratch_level), nodesPerFace, nDim);
    SharedMemView<double *> ws_scalarQNp1(team.thread_scratch(scratch_level), nodesPerFace);
    SharedMemView<double *> ws_bcScalarQ(team.thread_scratch(scratch_level), nodesPerFace);
    SharedMemView<double *> coordBip(team.thread_scratch(scratch_level), nDim);
    SharedMemView<int *> face_node_ordinal_vec(team.thread_scratch(scratch_level), nodesPerFace);

    // shape functions
    SharedMemView<double **> ws_face_shape_function(team.team_scratch(scratch_level), numScsBip, nodesPerFace);
    Kokkos::single(Kokkos::PerTeam(team), [&]() {
      meFC->shape_fcn(&ws_face_shape_function(0, 0));
    });
    team.team_barrier();

    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, length), [&](const size_t k) {
      // zero lhs/rhs
      for ( int p = 0; p < lhsSize; ++p )
        lhs[p] = 0.0;
      for ( int p = 0; p < rhsSize; ++p )
        rhs[p] = 0.0;

      // get face
      stk::mesh::Entity face = b[k];

      // pointer to face data
      const double * mdot = stk::mesh::field_data(*openMassFlowRate_, face);

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
        ws_scalarQNp1[ni] = *stk::mesh::field_data(scalarQNp1, node);
        ws_bcScalarQ[ni] = *stk::mesh::field_data(*bcScalarQ_, node);

        // gather vectors
        double * coords = stk::mesh::field_data(*coordinates_, node);
        for ( int i=0; i < nDim; ++i ) {
          ws_face_coordinates(ni, i) = coords[i];
        }
      }

      // extract the connected element to this exposed face; should be single in size!
      const stk::mesh::Entity* face_elem_rels = bulk_data.begin_elements(face);
      //ThrowAssert( bulk_data.num_elements(face) == 1 );

      // get element; its face ordinal number and populate face_node_ordinal_vec
      stk::mesh::Entity element = face_elem_rels[0];
      const int face_ordinal = bulk_data.begin_element_ordinals(face)[0];
      theElemTopo.side_node_ordinals(face_ordinal, &face_node_ordinal_vec(0));

      // mapping from ip to nodes for this ordinal
      const int *ipNodeMap = meSCS->ipNodeMap(face_ordinal);

      //==========================================
      // gather nodal data off of element; n/a
      //==========================================
      stk::mesh::Entity const * elem_node_rels = bulk_data.begin_nodes(element);
      int num_nodes = bulk_data.num_nodes(element);
      // sanity check on num nodes
      //ThrowAssert( num_nodes == nodesPerElement );
      for ( int ni = 0; ni < num_nodes; ++ni ) {
        // set connected nodes
        connected_nodes[ni] = elem_node_rels[ni];
      }

      // loop over face nodes
      for ( int ip = 0; ip < numScsBip; ++ip ) {

        const int opposingNode = meSCS->opposingNodes(face_ordinal,ip);
        const int nearestNode = ipNodeMap[ip];

        const int offSetSF_face = ip*nodesPerFace;

        // left and right nodes; right is on the face; left is the opposing node
        stk::mesh::Entity nodeL = elem_node_rels[opposingNode];
        stk::mesh::Entity nodeR = elem_node_rels[nearestNode];

        // zero out vector quantities
        for ( int j = 0; j < nDim; ++j )
          coordBip[j] = 0.0;

        // interpolate to bip
        double qIp = 0.0;
        double qIpEntrain = 0.0;
        for ( int ic = 0; ic < nodesPerFace; ++ic ) {
          const double r = ws_face_shape_function(ip, ic);
          qIp += r*ws_scalarQNp1[ic];
          qIpEntrain += r*ws_bcScalarQ[ic];
          const int offSetFN = ic*nDim;
          for ( int j = 0; j < nDim; ++j ) {
            coordBip[j] += r*ws_face_coordinates(ic, j);
          }
        }

        // Peclet factor; along the edge is fine
        const double densL       = *stk::mesh::field_data(densityNp1, nodeL);
        const double densR       = *stk::mesh::field_data(densityNp1, nodeR);
        const double diffCoeffL  = *stk::mesh::field_data(*diffFluxCoeff_, nodeL);
        const double diffCoeffR  = *stk::mesh::field_data(*diffFluxCoeff_, nodeR);
        const double scalarQNp1R = *stk::mesh::field_data(scalarQNp1, nodeR);
        const double *vrtmL      =  stk::mesh::field_data(*velocityRTM_, nodeL);
        const double *vrtmR      =  stk::mesh::field_data(*velocityRTM_, nodeR);
        const double *coordL     =  stk::mesh::field_data(*coordinates_, nodeL);
        const double *coordR     =  stk::mesh::field_data(*coordinates_, nodeR);
        const double *dqdxR      =  stk::mesh::field_data(*dqdx_, nodeR);

        double udotx = 0.0;
        double dqR = 0.0;
        for ( int i = 0; i < nDim; ++i ) {
          const double dxi = coordR[i]  - coordL[i];
          udotx += 0.5*dxi*(vrtmL[i] + vrtmR[i]);
          // extrapolation
          const double dx_bip = coordBip[i] - coordR[i];
          dqR += dx_bip*dqdxR[i]*hoUpwind;
        }
        const double qIpUpw = scalarQNp1R + dqR;

        const double diffIp = 0.5*(diffCoeffL/densL + diffCoeffR/densR);
        const double pecfac = pecletFunction_->execute(std::abs(udotx)/(diffIp+small));
        const double om_pecfac = 1.0-pecfac;

        //================================
        // advection first (and only)
        //================================
        const double tmdot = mdot[ip];

        const int rowR = nearestNode*nodesPerElement;

        // advection; leaving the domain
        if ( tmdot > 0.0 ) {

          // central; is simply qIp

          // upwind
          const double qUpwind = alphaUpw*qIpUpw + (om_alphaUpw)*qIp;

          // total advection
          const double aflux = tmdot*(pecfac*qUpwind+om_pecfac*qIp);

          rhs[nearestNode] -= aflux;

          // upwind lhs
          lhs[rowR+nearestNode] += tmdot*pecfac*alphaUpw;

          // central part
          const double fac = tmdot*(pecfac*om_alphaUpw+om_pecfac);
          for ( int ic = 0; ic < nodesPerFace; ++ic ) {
            const double r = ws_face_shape_function(ip, ic);
            const int nn = face_node_ordinal_vec[ic];
            lhs[rowR+nn] += r*fac;
          }
        }
        else {

          // extrainment; advect in from specified value
          const double aflux = tmdot*qIpEntrain;
          rhs[nearestNode] -= aflux;
        }
      }

      apply_coeff(connected_nodes, rhs, lhs, localIdsScratch, __FILE__);
    });
  });
}

} // namespace nalu
} // namespace Sierra
