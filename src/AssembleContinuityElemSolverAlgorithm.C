/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


// nalu
#include <AssembleContinuityElemSolverAlgorithm.h>
#include <EquationSystem.h>
#include <SolverAlgorithm.h>

#include <FieldTypeDef.h>
#include <LinearSystem.h>
#include <Realm.h>
#include <SupplementalAlgorithm.h>
#include <master_element/MasterElement.h>

// stk_mesh/base/fem
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Field.hpp>
#include <stk_mesh/base/GetBuckets.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/Part.hpp>

namespace sierra{
namespace nalu{

//==========================================================================
// Class Definition
//==========================================================================
// AssembleContinuityElemSolverAlgorithm - add LHS/RHS for continuity
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
AssembleContinuityElemSolverAlgorithm::AssembleContinuityElemSolverAlgorithm(
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
    shiftMdot_(realm_.get_cvfem_shifted_mdot()),
    shiftPoisson_(realm_.get_cvfem_shifted_poisson()),
    reducedSensitivities_(realm_.get_cvfem_reduced_sens_poisson())
{
  // extract fields; nodal
  stk::mesh::MetaData & meta_data = realm_.meta_data();
  if ( meshMotion_ )
    velocityRTM_ = meta_data.get_field<VectorFieldType>(stk::topology::NODE_RANK, "velocity_rtm");
  else
    velocityRTM_ = meta_data.get_field<VectorFieldType>(stk::topology::NODE_RANK, "velocity");
  Gpdx_ = meta_data.get_field<VectorFieldType>(stk::topology::NODE_RANK, "dpdx");
  coordinates_ = meta_data.get_field<VectorFieldType>(stk::topology::NODE_RANK, realm_.get_coordinates_name());
  pressure_ = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "pressure");
  density_ = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "density");

  // Implementation details: code is designed to manage the following
  // When shiftPoisson_ is TRUE, reducedSensitivities_ is enforced to be TRUE
  // However, shiftPoisson_ can be FALSE while reducedSensitivities_ is TRUE
}

//--------------------------------------------------------------------------
//-------- initialize_connectivity -----------------------------------------
//--------------------------------------------------------------------------
void
AssembleContinuityElemSolverAlgorithm::initialize_connectivity()
{
  eqSystem_->linsys_->buildElemToNodeGraph(partVec_);
}

//--------------------------------------------------------------------------
//-------- execute ---------------------------------------------------------
//--------------------------------------------------------------------------
void
AssembleContinuityElemSolverAlgorithm::execute()
{

  stk::mesh::MetaData & meta_data = realm_.meta_data();

  const int nDim = meta_data.spatial_dimension();

  const int maxElementsPerBucket = 512;
  const int maxNodesPerElement = 8;
  const int maxNumScsIp = 16;
  const int maxlhsSize = maxNodesPerElement*maxNodesPerElement;
  const int maxrhsSize = maxNodesPerElement;

  // time step
  const double dt = realm_.get_time_step();
  const double gamma1 = realm_.get_gamma1();
  const double projTimeScale = dt/gamma1;

  // deal with interpolation procedure
  const double interpTogether = realm_.get_mdot_interp();
  const double om_interpTogether = 1.0-interpTogether;

  // space for LHS/RHS; nodesPerElem*nodesPerElem and nodesPerElem
  std::vector<double> lhs;
  std::vector<double> rhs;
  std::vector<stk::mesh::Entity> connected_nodes;

  // supplemental algorithm setup
  const size_t supplementalAlgSize = supplementalAlg_.size();
  for ( size_t i = 0; i < supplementalAlgSize; ++i )
    supplementalAlg_[i]->setup();


  // deal with state
  ScalarFieldType &densityNp1 = density_->field_of_state(stk::mesh::StateNP1);

  // define some common selectors
  stk::mesh::Selector s_locally_owned_union = meta_data.locally_owned_part()
    & stk::mesh::selectUnion(partVec_) 
    & !(realm_.get_inactive_selector());

  stk::mesh::BucketVector const& elem_buckets =
    realm_.get_buckets( stk::topology::ELEMENT_RANK, s_locally_owned_union );

  const int bytes_per_team = SharedMemView<double *>::shmem_size(maxNumScsIp * maxNodesPerElement);
  // TODO: This may substantially overestimate the scratch space needed depending on what
  // element types are actually present. We should investigate whether the cost of this matters
  // and if so consider the Aria approach where a separate algorithm is created per topology.
  const int bytes_per_thread =
      SharedMemView<double *>::shmem_size(maxNodesPerElement*nDim) + //ws_vrtm
      SharedMemView<double *>::shmem_size(maxNodesPerElement*nDim) + //ws_Gpdx
      SharedMemView<double *>::shmem_size(maxNodesPerElement*nDim) + //ws_coordinates
      SharedMemView<double *>::shmem_size(maxNodesPerElement) + //ws_pressure
      SharedMemView<double *>::shmem_size(maxNodesPerElement) + //ws_density
      SharedMemView<double *>::shmem_size(maxNumScsIp*nDim) + //ws_scs_areav
      SharedMemView<double *>::shmem_size(nDim*maxNumScsIp*maxNodesPerElement) + //ws_dndx
      SharedMemView<double *>::shmem_size(nDim*maxNumScsIp*maxNodesPerElement) + //ws_dndx_lhs
      SharedMemView<double *>::shmem_size(nDim*maxNumScsIp*maxNodesPerElement) + //ws_deriv
      SharedMemView<double *>::shmem_size(maxNumScsIp) + //ws_detj
      SharedMemView<double *>::shmem_size(nDim) + //uIp
      SharedMemView<double *>::shmem_size(nDim) + //rho_uIp
      SharedMemView<double *>::shmem_size(nDim) + //GpdxIp
      SharedMemView<double *>::shmem_size(nDim) + //dpdxIp
      SharedMemView<double *>::shmem_size(maxlhsSize) +
      SharedMemView<double *>::shmem_size(maxrhsSize) +
      SharedMemView<stk::mesh::Entity *>::shmem_size(maxNodesPerElement) +
      SharedMemView<int *>::shmem_size(maxrhsSize); // For TpetraLinearSystem::sumInto vector of localIds

  auto team_exec = get_team_policy(elem_buckets.size(), bytes_per_team, bytes_per_thread);

  Kokkos::parallel_for("Nalu::AssembleContinuityElemSolverAlgorithm::execute",
      team_exec, [&] (const DeviceTeam & team)
  {
    const int ib = team.league_rank();
    stk::mesh::Bucket & b = *elem_buckets[ib];
    const stk::mesh::Bucket::size_type length   = b.size();

    // extract master element
    MasterElement *meSCS = realm_.get_surface_master_element(b.topology());
    MasterElement *meSCV = realm_.get_volume_master_element(b.topology());

    // extract master element specifics
    const int nodesPerElement = meSCS->nodesPerElement_;
    const int numScsIp = meSCS->numIntPoints_;
    const int *lrscv = meSCS->adjacentNodes();

    // resize some things; matrix related
    const int lhsSize = nodesPerElement*nodesPerElement;
    const int rhsSize = nodesPerElement;

    SharedMemView<double**> ws_shape_function_(team.team_shmem(), numScsIp, nodesPerElement);

    // These are the per-thread handles. Better interface being worked on by Kokkos.
    SharedMemView<stk::mesh::Entity*> connected_nodes;
    SharedMemView<double*> lhs;
    SharedMemView<double*> rhs;
    SharedMemView<int*> localIdsScratch;
    SharedMemView<double*> ws_vrtm;
    SharedMemView<double*> ws_Gpdx;
    SharedMemView<double*> ws_coordinates;
    SharedMemView<double*> ws_pressure;
    SharedMemView<double*> ws_density;
    SharedMemView<double*> ws_scs_areav;
    SharedMemView<double*> ws_dndx;
    SharedMemView<double*> ws_dndx_lhs;
    SharedMemView<double*> ws_deriv;
    SharedMemView<double*> ws_det_j;
    SharedMemView<double*> uIp(team.team_shmem(), nDim);
    SharedMemView<double*> rho_uIp(team.team_shmem(), nDim);
    SharedMemView<double*> GpdxIp(team.team_shmem(), nDim);
    SharedMemView<double*> dpdxIp(team.team_shmem(), nDim);
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
          SharedMemView<double**>(team.team_shmem(), team.team_size(), nodesPerElement*nDim),
          team.team_rank(), Kokkos::ALL());
      ws_Gpdx = Kokkos::subview(
          SharedMemView<double**>(team.team_shmem(), team.team_size(), nodesPerElement*nDim),
          team.team_rank(), Kokkos::ALL());
      ws_coordinates = Kokkos::subview(
          SharedMemView<double**>(team.team_shmem(), team.team_size(), nodesPerElement*nDim),
          team.team_rank(), Kokkos::ALL());
      ws_pressure = Kokkos::subview(
          SharedMemView<double**>(team.team_shmem(), team.team_size(), nodesPerElement),
          team.team_rank(), Kokkos::ALL());
      ws_density = Kokkos::subview(
          SharedMemView<double**>(team.team_shmem(), team.team_size(), nodesPerElement),
          team.team_rank(), Kokkos::ALL());
      ws_scs_areav = Kokkos::subview(
          SharedMemView<double**>(team.team_shmem(), team.team_size(), numScsIp*nDim),
          team.team_rank(), Kokkos::ALL());
      ws_dndx = Kokkos::subview(
          SharedMemView<double**>(team.team_shmem(), team.team_size(), numScsIp*nDim*nodesPerElement),
          team.team_rank(), Kokkos::ALL());
      ws_dndx_lhs = Kokkos::subview(
          SharedMemView<double**>(team.team_shmem(), team.team_size(), numScsIp*nDim*nodesPerElement),
          team.team_rank(), Kokkos::ALL());
      ws_deriv = Kokkos::subview(
          SharedMemView<double**>(team.team_shmem(), team.team_size(), numScsIp*nDim*nodesPerElement),
          team.team_rank(), Kokkos::ALL());
      ws_det_j = Kokkos::subview(
          SharedMemView<double**>(team.team_shmem(), team.team_size(), numScsIp),
          team.team_rank(), Kokkos::ALL());
        localIdsScratch = Kokkos::subview(
            SharedMemView<int**> (team.team_shmem(), team.team_size(), rhsSize),
            team.team_rank(), Kokkos::ALL());

    }


    Kokkos::single(Kokkos::PerTeam(team), [&]()
    {
      if ( shiftMdot_)
        meSCS->shifted_shape_fcn(&ws_shape_function_(0, 0));
      else
        meSCS->shape_fcn(&ws_shape_function_(0, 0));

      // resize possible supplemental element alg
      for ( size_t i = 0; i < supplementalAlgSize; ++i )
        supplementalAlg_[i]->elem_resize(meSCS, meSCV);
    });

    team.team_barrier();

    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, length), [&] (const size_t k)
    {
      const stk::mesh::Entity elem = b[k];
      // zero lhs/rhs
      for ( int p = 0; p < lhsSize; ++p )
        lhs[p] = 0.0;
      for ( int p = 0; p < rhsSize; ++p )
        rhs[p] = 0.0;

      //===============================================
      // gather nodal data; this is how we do it now..
      //===============================================
      stk::mesh::Entity const *  node_rels = b.begin_nodes(k);
      int num_nodes = b.num_nodes(k);

      for ( int ni = 0; ni < num_nodes; ++ni ) {
        stk::mesh::Entity node = node_rels[ni];

        // set connected nodes
        connected_nodes[ni] = node;

        // pointers to real data
        const double * Gjp    = stk::mesh::field_data(*Gpdx_, node );
        const double * coords = stk::mesh::field_data(*coordinates_, node );
        const double * vrtm   = stk::mesh::field_data(*velocityRTM_, node );

        // gather scalars
        ws_pressure[ni] = *stk::mesh::field_data(*pressure_, node );
        ws_density[ni]  = *stk::mesh::field_data(densityNp1, node );

        // gather vectors
        const int niNdim = ni*nDim;
        for ( int j=0; j < nDim; ++j ) {
          ws_vrtm[niNdim+j] = vrtm[j];
          ws_Gpdx[niNdim+j] = Gjp[j];
          ws_coordinates[niNdim+j] = coords[j];
        }
      }

      // compute geometry
      double scs_error = 0.0;
      meSCS->determinant(1, &ws_coordinates[0], &ws_scs_areav[0], &scs_error);

      // compute dndx for residual
      if ( shiftPoisson_ )
        meSCS->shifted_grad_op(1, &ws_coordinates[0], &ws_dndx[0], &ws_deriv[0], &ws_det_j[0], &scs_error);
      else
        meSCS->grad_op(1, &ws_coordinates[0], &ws_dndx[0], &ws_deriv[0], &ws_det_j[0], &scs_error);

      // compute dndx for LHS
      auto p_dndx_lhs = shiftPoisson_ ? &ws_dndx[0] : reducedSensitivities_ ? &ws_dndx_lhs[0] : &ws_dndx[0];
      if ( !shiftPoisson_ && reducedSensitivities_ )
        meSCS->shifted_grad_op(1, &ws_coordinates[0], &p_dndx_lhs[0], &ws_deriv[0], &ws_det_j[0], &scs_error);

      for ( int ip = 0; ip < numScsIp; ++ip ) {

        // left and right nodes for this ip
        const int il = lrscv[2*ip];
        const int ir = lrscv[2*ip+1];

        // corresponding matrix rows
        int rowL = il*nodesPerElement;
        int rowR = ir*nodesPerElement;

        // setup for ip values; sneak in geometry for possible reduced sens
        for ( int j = 0; j < nDim; ++j ) {
          uIp[j] = 0.0;
          rho_uIp[j] = 0.0;
          GpdxIp[j] = 0.0;
          dpdxIp[j] = 0.0;
        }
        double rhoIp = 0.0;

        for ( int ic = 0; ic < nodesPerElement; ++ic ) {

          const double r = ws_shape_function_(ip, ic);
          const double nodalPressure = ws_pressure[ic];
          const double nodalRho = ws_density[ic];

          rhoIp += r*nodalRho;

          double lhsfac = 0.0;
          const int offSetDnDx = nDim*nodesPerElement*ip + ic*nDim;
          for ( int j = 0; j < nDim; ++j ) {
            GpdxIp[j] += r*ws_Gpdx[nDim*ic+j];
            uIp[j] += r*ws_vrtm[nDim*ic+j];
            rho_uIp[j] += r*nodalRho*ws_vrtm[nDim*ic+j];
            dpdxIp[j] += ws_dndx[offSetDnDx+j]*nodalPressure;
            lhsfac += -p_dndx_lhs[offSetDnDx+j]*ws_scs_areav[ip*nDim+j];
          }

          // assemble to lhs; left
          lhs[rowL+ic] += lhsfac;

          // assemble to lhs; right
          lhs[rowR+ic] -= lhsfac;

        }

        // assemble mdot
        double mdot = 0.0;
        for ( int j = 0; j < nDim; ++j ) {
          mdot += (interpTogether*rho_uIp[j] + om_interpTogether*rhoIp*uIp[j]
                                                                               - projTimeScale*(dpdxIp[j] - GpdxIp[j]))*ws_scs_areav[ip*nDim+j];
        }

        // residual; left and right
        rhs[il] -= mdot/projTimeScale;
        rhs[ir] += mdot/projTimeScale;
      }

      // call supplemental
      for ( size_t i = 0; i < supplementalAlgSize; ++i )
        supplementalAlg_[i]->elem_execute( &lhs[0], &rhs[0], elem, meSCS, meSCV);

      apply_coeff(connected_nodes, rhs, lhs, localIdsScratch, __FILE__);
    });
  });
}

} // namespace nalu
} // namespace Sierra
