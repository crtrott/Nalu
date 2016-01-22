/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


// nalu
#include <AssembleScalarElemSolverAlgorithm.h>
#include <EquationSystem.h>
#include <SolverAlgorithm.h>

#include <FieldTypeDef.h>
#include <LinearSystem.h>
#include <PecletFunction.h>
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
// AssembleScalarElemSolverAlgorithm - add LHS/RHS for scalar
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
AssembleScalarElemSolverAlgorithm::AssembleScalarElemSolverAlgorithm(
  Realm &realm,
  stk::mesh::Part *part,
  EquationSystem *eqSystem,
  ScalarFieldType *scalarQ,
  VectorFieldType *dqdx,
  ScalarFieldType *diffFluxCoeff)
  : SolverAlgorithm(realm, part, eqSystem),
    meshMotion_(realm_.does_mesh_move()),
    scalarQ_(scalarQ),
    dqdx_(dqdx),
    diffFluxCoeff_(diffFluxCoeff),
    velocityRTM_(NULL),
    coordinates_(NULL),
    density_(NULL),
    massFlowRate_(NULL),
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
  massFlowRate_ = meta_data.get_field<GenericFieldType>(stk::topology::ELEMENT_RANK, "mass_flow_rate_scs");

  // create the peclet blending function
  pecletFunction_ = eqSystem->create_peclet_function(scalarQ_->name());
  
  /* Notes:

  Matrix layout is in row major. For a npe = 4 (quad) and nDof = 1:

  RHS = (resQ0, resQ1, resQ2, resQ3)

  The LHS is, therefore,

  row 0: d/dQ0(ResQ0), ., ., ., .,  d/dQ3(ResQ0)
  row 1: d/dQ0(ResQ1), ., ., ., .,  d/dQ3(ResQ1)

  */
}

//--------------------------------------------------------------------------
//-------- destructor ------------------------------------------------------
//--------------------------------------------------------------------------
AssembleScalarElemSolverAlgorithm::~AssembleScalarElemSolverAlgorithm()
{
  delete pecletFunction_;
}
                                                                     
//--------------------------------------------------------------------------
//-------- initialize_connectivity -----------------------------------------
//--------------------------------------------------------------------------
void
AssembleScalarElemSolverAlgorithm::initialize_connectivity()
{
  eqSystem_->linsys_->buildElemToNodeGraph(partVec_);
}

//--------------------------------------------------------------------------
//-------- execute ---------------------------------------------------------
//--------------------------------------------------------------------------
void
AssembleScalarElemSolverAlgorithm::execute()
{

  stk::mesh::BulkData & bulk_data = realm_.bulk_data();
  stk::mesh::MetaData & meta_data = realm_.meta_data();

  const int nDim = meta_data.spatial_dimension();
  const double small = 1.0e-16;

  // extract user advection options (allow to potentially change over time)
  const std::string dofName = scalarQ_->name();
  const double alpha = realm_.get_alpha_factor(dofName);
  const double alphaUpw = realm_.get_alpha_upw_factor(dofName);
  const double hoUpwind = realm_.get_upw_factor(dofName);
  const bool useLimiter = realm_.primitive_uses_limiter(dofName);

  // one minus flavor..
  const double om_alpha = 1.0-alpha;
  const double om_alphaUpw = 1.0-alphaUpw;

  // space for LHS/RHS; nodesPerElem*nodesPerElem* and nodesPerElem

  // supplemental algorithm setup
  const size_t supplementalAlgSize = supplementalAlg_.size();
  for ( size_t i = 0; i < supplementalAlgSize; ++i )
    supplementalAlg_[i]->setup();


  const int maxNodesPerElement = 8;
  const int maxNumScsIp = 16;
  const int maxLhsSize = maxNodesPerElement*maxNodesPerElement;
  const int maxRhsSize = maxNodesPerElement;
  const int bytes_per_team = SharedMemView<double *>::shmem_size(maxNumScsIp * maxNodesPerElement);
  // TODO: This may substantially overestimate the scratch space needed depending on what
  // element types are actually present. We should investigate whether the cost of this matters
  // and if so consider the Aria approach where a separate algorithm is created per topology.
  const int bytes_per_thread =
      SharedMemView<double *>::shmem_size(maxLhsSize) + //lhs
      SharedMemView<double *>::shmem_size(maxRhsSize) + //rhs
      SharedMemView<double *>::shmem_size(maxNodesPerElement * nDim) + //ws_vrtm
      SharedMemView<double *>::shmem_size(maxNodesPerElement * nDim) + //ws_coord
      SharedMemView<double *>::shmem_size(maxNodesPerElement) + //ws_scalarQNp1
      SharedMemView<double *>::shmem_size(maxNodesPerElement * nDim) + //ws_dqdx
      SharedMemView<double *>::shmem_size(maxNodesPerElement) + //ws_density
      SharedMemView<double *>::shmem_size(maxNodesPerElement) + //ws_diffFluxCoeff
      SharedMemView<double *>::shmem_size(maxNumScsIp * nDim) + //ws_scs_area_v
      SharedMemView<double *>::shmem_size(maxNumScsIp * nDim * maxNodesPerElement) + //ws_dndx
      SharedMemView<double *>::shmem_size(maxNumScsIp * nDim * maxNodesPerElement) + //ws_deriv
      SharedMemView<double *>::shmem_size(maxNumScsIp) + //ws_detj
      SharedMemView<double *>::shmem_size(nDim) + //coordIp
      SharedMemView<stk::mesh::Entity *>::shmem_size(maxNodesPerElement) + //elements
      SharedMemView<int *>::shmem_size(maxRhsSize); // For TpetraLinearSystem::sumInto vector of localIds

  // deal with state
  ScalarFieldType &scalarQNp1   = scalarQ_->field_of_state(stk::mesh::StateNP1);
  ScalarFieldType &densityNp1 = density_->field_of_state(stk::mesh::StateNP1);

  // define some common selectors
  stk::mesh::Selector s_locally_owned_union = meta_data.locally_owned_part()
    & stk::mesh::selectUnion(partVec_) 
    & !(realm_.get_inactive_selector());

  stk::mesh::BucketVector const& elem_buckets =
    realm_.get_buckets( stk::topology::ELEMENT_RANK, s_locally_owned_union );
  auto team_exec = get_team_policy(elem_buckets.size(), bytes_per_team, bytes_per_thread);

  Kokkos::parallel_for("Nalu::AssembleScalarElemDiffSolverAlgorithm::execute",
      team_exec, [&] (const DeviceTeam & team) {
      const int ib = team.league_rank();
      stk::mesh::Bucket & b = *elem_buckets[ib];
      const stk::mesh::Bucket::size_type length = b.size();

      MasterElement *meSCS = realm_.get_surface_master_element(b.topology());
      MasterElement *meSCV = realm_.get_volume_master_element(b.topology());

      const int nDim_ = nDim;
      const int nodesPerElement = meSCS->nodesPerElement_;
      const int numScsIp = meSCS->numIntPoints_;
      const int rhsSize = nodesPerElement;
      const int lhsSize = nodesPerElement*nodesPerElement;

      SharedMemView<double**> ws_shape_function(team.team_shmem(), numScsIp, nodesPerElement);

      SharedMemView<double*> lhs;
      SharedMemView<double*> rhs;
      SharedMemView<stk::mesh::Entity *> connected_nodes;
      SharedMemView<double*> ws_vrtm;
      SharedMemView<double*> ws_coordinates;
      SharedMemView<double*> ws_scalarQNp1;
      SharedMemView<double*> ws_dqdx;
      SharedMemView<double*> ws_density;
      SharedMemView<double*> ws_diffFluxCoeff;
      SharedMemView<double*> ws_scs_areav;
      SharedMemView<double*> ws_dndx;
      SharedMemView<double*> ws_deriv;
      SharedMemView<double*> ws_det_j;
      SharedMemView<double*> coordIp;
      SharedMemView<int*> localIdsScratch;
      {
        connected_nodes = Kokkos::subview(
            SharedMemView<stk::mesh::Entity**> (team.team_shmem(), team.team_size(), nodesPerElement),
            team.team_rank(), Kokkos::ALL());
        lhs = Kokkos::subview(
            SharedMemView<double**>(team.team_shmem(), team.team_size(), lhsSize),
            team.team_rank(), Kokkos::ALL());
        rhs = Kokkos::subview(
            SharedMemView<double**>(team.team_shmem(), team.team_size(), rhsSize),
            team.team_rank(), Kokkos::ALL());
        ws_vrtm = Kokkos::subview(
            SharedMemView<double**> (team.team_shmem(), team.team_size(), nodesPerElement*nDim),
            team.team_rank(), Kokkos::ALL());
        ws_coordinates = Kokkos::subview(
            SharedMemView<double**> (team.team_shmem(), team.team_size(), nodesPerElement*nDim),
            team.team_rank(), Kokkos::ALL());
        ws_scalarQNp1 = Kokkos::subview(
            SharedMemView<double**> (team.team_shmem(), team.team_size(), nodesPerElement),
            team.team_rank(), Kokkos::ALL());
        ws_dqdx = Kokkos::subview(
            SharedMemView<double**> (team.team_shmem(), team.team_size(), nodesPerElement*nDim_),
            team.team_rank(), Kokkos::ALL());
        ws_density = Kokkos::subview(
            SharedMemView<double**> (team.team_shmem(), team.team_size(), nodesPerElement),
            team.team_rank(), Kokkos::ALL());
        ws_diffFluxCoeff = Kokkos::subview(
            SharedMemView<double**> (team.team_shmem(), team.team_size(), nodesPerElement),
            team.team_rank(), Kokkos::ALL());
        ws_scs_areav = Kokkos::subview(
            SharedMemView<double**> (team.team_shmem(), team.team_size(), numScsIp*nDim),
            team.team_rank(), Kokkos::ALL());
        ws_dndx = Kokkos::subview(
            SharedMemView<double**> (team.team_shmem(), team.team_size(), numScsIp*nodesPerElement*nDim),
            team.team_rank(), Kokkos::ALL());
        ws_deriv = Kokkos::subview(
            SharedMemView<double**> (team.team_shmem(), team.team_size(), numScsIp*nodesPerElement*nDim),
            team.team_rank(), Kokkos::ALL());
        ws_det_j = Kokkos::subview(
            SharedMemView<double**> (team.team_shmem(), team.team_size(), numScsIp),
            team.team_rank(), Kokkos::ALL());
        localIdsScratch = Kokkos::subview(
            SharedMemView<int**> (team.team_shmem(), team.team_size(), rhsSize),
            team.team_rank(), Kokkos::ALL());
        coordIp = Kokkos::subview(
            SharedMemView<double**> (team.team_shmem(), team.team_size(), nDim),
            team.team_rank(), Kokkos::ALL());
      }

      Kokkos::single(Kokkos::PerTeam(team), [&]() {
        meSCS->shape_fcn(&ws_shape_function(0, 0));
        for ( size_t i = 0; i < supplementalAlgSize; ++i )
          supplementalAlg_[i]->elem_resize(meSCS, meSCV);
      });
      team.team_barrier();
      auto lrscv = meSCS->adjacentNodes();

      Kokkos::parallel_for(Kokkos::TeamThreadRange(team, length), [&] (const size_t k) {
        const stk::mesh::Entity elem = b[k];

        // zero lhs/rhs
        for ( int p = 0; p < lhsSize; ++p )
          lhs[p] = 0.0;
        for ( int p = 0; p < rhsSize; ++p )
          rhs[p] = 0.0;


        // ip data for this element; scs and scv
        const double *mdot = stk::mesh::field_data(*massFlowRate_, elem );

        //===============================================
        // gather nodal data; this is how we do it now..
        //===============================================
        stk::mesh::Entity const * node_rels = bulk_data.begin_nodes(elem);
        int num_nodes = bulk_data.num_nodes(elem);

        // sanity check on num nodes
        ThrowAssert( num_nodes == nodesPerElement );

        for ( int ni = 0; ni < num_nodes; ++ni ) {
          stk::mesh::Entity node = node_rels[ni];

          // set connected nodes
          connected_nodes[ni] = node;

          // pointers to real data
          const double * vrtm   = stk::mesh::field_data(*velocityRTM_, node );
          const double * coords = stk::mesh::field_data(*coordinates_, node );
          const double * dq     = stk::mesh::field_data(*dqdx_, node );

          // gather scalars
          ws_scalarQNp1[ni]    = *stk::mesh::field_data(scalarQNp1, node );
          ws_density[ni]       = *stk::mesh::field_data(densityNp1, node );
          ws_diffFluxCoeff[ni] = *stk::mesh::field_data(*diffFluxCoeff_, node );

          // gather vectors
          const int niNdim = ni*nDim;
          for ( int i=0; i < nDim; ++i ) {
            ws_vrtm[niNdim+i] = vrtm[i];
            ws_coordinates[niNdim+i] = coords[i];
            ws_dqdx[niNdim+i] = dq[i];
          }
        }

        // compute geometry
        double scs_error = 0.0;
        meSCS->determinant(1, &ws_coordinates[0], &ws_scs_areav[0], &scs_error);

        // compute dndx
        meSCS->grad_op(1, &ws_coordinates[0], &ws_dndx[0], &ws_deriv[0], &ws_det_j[0], &scs_error);

        for ( int ip = 0; ip < numScsIp; ++ip ) {

          // left and right nodes for this ip
          const int il = lrscv[2*ip];
          const int ir = lrscv[2*ip+1];

          // corresponding matrix rows
          const int rowL = il*nodesPerElement;
          const int rowR = ir*nodesPerElement;

          // save off mdot
          const double tmdot = mdot[ip];

          // zero out values of interest for this ip
          for ( int j = 0; j < nDim; ++j ) {
            coordIp[j] = 0.0;
          }

          // save off ip values; offset to Shape Function
          double rhoIp = 0.0;
          double muIp = 0.0;
          double qIp = 0.0;
          for ( int ic = 0; ic < nodesPerElement; ++ic ) {
            const double r = ws_shape_function(ip, ic);
            rhoIp += r*ws_density[ic];
            muIp += r*ws_diffFluxCoeff[ic];
            qIp += r*ws_scalarQNp1[ic];
            // compute scs point values
            for ( int i = 0; i < nDim; ++i ) {
              coordIp[i] += r*ws_coordinates[ic*nDim+i];
            }
          }

          // Peclet factor; along the edge
          const double diffIp = 0.5*(ws_diffFluxCoeff[il]/ws_density[il]
                                                                   + ws_diffFluxCoeff[ir]/ws_density[ir]);
          double udotx = 0.0;
          for(int j = 0; j < nDim; ++j ) {
            const double dxj = ws_coordinates[ir*nDim+j]-ws_coordinates[il*nDim+j];
            const double uj = 0.5*(ws_vrtm[il*nDim+j] + ws_vrtm[ir*nDim+j]);
            udotx += uj*dxj;
          }
          const double pecfac = pecletFunction_->execute(std::abs(udotx)/(diffIp+small));
          const double om_pecfac = 1.0-pecfac;

          // left and right extrapolation
          double dqL = 0.0;
          double dqR = 0.0;
          for(int j = 0; j < nDim; ++j ) {
            const double dxjL = coordIp[j] - ws_coordinates[il*nDim+j];
            const double dxjR = ws_coordinates[ir*nDim+j] - coordIp[j];
            dqL += dxjL*ws_dqdx[nDim*il+j];
            dqR += dxjR*ws_dqdx[nDim*ir+j];
          }

          // add limiter if appropriate
          double limitL = 1.0;
          double limitR = 1.0;
          if ( useLimiter ) {
            const double dq = ws_scalarQNp1[ir] - ws_scalarQNp1[il];
            const double dqMl = 2.0*2.0*dqL - dq;
            const double dqMr = 2.0*2.0*dqR - dq;
            limitL = van_leer(dqMl, dq, small);
            limitR = van_leer(dqMr, dq, small);
          }

          // extrapolated; for now limit (along edge is fine)
          const double qIpL = ws_scalarQNp1[il] + dqL*hoUpwind*limitL;
          const double qIpR = ws_scalarQNp1[ir] - dqR*hoUpwind*limitR;

          // assemble advection; rhs and upwind contributions

          // 2nd order central; simply qIp from above

          // upwind
          const double qUpwind = (tmdot > 0) ? alphaUpw*qIpL + om_alphaUpw*qIp
              : alphaUpw*qIpR + om_alphaUpw*qIp;

          // generalized central (2nd and 4th order)
          const double qHatL = alpha*qIpL + om_alpha*qIp;
          const double qHatR = alpha*qIpR + om_alpha*qIp;
          const double qCds = 0.5*(qHatL + qHatR);

          // total advection
          const double aflux = tmdot*(pecfac*qUpwind + om_pecfac*qCds);

          // right hand side; L and R
          rhs[il] -= aflux;
          rhs[ir] += aflux;

          // advection operator sens; all but central

          // upwind advection (includes 4th); left node
          const double alhsfacL = 0.5*(tmdot+std::abs(tmdot))*pecfac*alphaUpw
              + 0.5*alpha*om_pecfac*tmdot;
          lhs[rowL+il] += alhsfacL;
          lhs[rowR+il] -= alhsfacL;

          // upwind advection; right node
          const double alhsfacR = 0.5*(tmdot-std::abs(tmdot))*pecfac*alphaUpw
              + 0.5*alpha*om_pecfac*tmdot;
          lhs[rowR+ir] -= alhsfacR;
          lhs[rowL+ir] += alhsfacR;

          double qDiff = 0.0;
          for ( int ic = 0; ic < nodesPerElement; ++ic ) {

            // shape function
            const double r = ws_shape_function(ip, ic);

            // upwind (il/ir) handled above; collect terms on alpha and alphaUpw
            const double lhsfacAdv = r*tmdot*(pecfac*om_alphaUpw + om_pecfac*om_alpha);

            // advection operator lhs; rhs handled above
            // lhs; il then ir
            lhs[rowL+ic] += lhsfacAdv;
            lhs[rowR+ic] -= lhsfacAdv;

            // diffusion
            double lhsfacDiff = 0.0;
            const int offSetDnDx = nDim*nodesPerElement*ip + ic*nDim;
            for ( int j = 0; j < nDim; ++j ) {
              lhsfacDiff += -muIp*ws_dndx[offSetDnDx+j]*ws_scs_areav[ip*nDim+j];
            }

            qDiff += lhsfacDiff*ws_scalarQNp1[ic];

            // lhs; il then ir
            lhs[rowL+ic] += lhsfacDiff;
            lhs[rowR+ic] -= lhsfacDiff;
          }

          // rhs; il then ir
          rhs[il] -= qDiff;
          rhs[ir] += qDiff;

        }

        // call supplemental
        for ( size_t i = 0; i < supplementalAlgSize; ++i )
          supplementalAlg_[i]->elem_execute( &lhs[0], &rhs[0], elem, meSCS, meSCV);

        apply_coeff(connected_nodes, rhs, lhs, localIdsScratch, __FILE__);
      });
  });
}

//--------------------------------------------------------------------------
//-------- van_leer ---------------------------------------------------------
//--------------------------------------------------------------------------
double
AssembleScalarElemSolverAlgorithm::van_leer(
  const double &dqm,
  const double &dqp,
  const double &small)
{
  double limit = (2.0*(dqm*dqp+std::abs(dqm*dqp))) /
    ((dqm+dqp)*(dqm+dqp)+small);
  return limit;
}

} // namespace nalu
} // namespace Sierra
