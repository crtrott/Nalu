/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


// nalu
#include <AssembleMomentumElemSolverAlgorithm.h>
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

#include <KokkosInterface.h>

namespace sierra{
namespace nalu{

//==========================================================================
// Class Definition
//==========================================================================
// AssembleMomentumElemSolverAlgorithm - add LHS/RHS for uvw momentum
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
AssembleMomentumElemSolverAlgorithm::AssembleMomentumElemSolverAlgorithm(
  Realm &realm,
  stk::mesh::Part *part,
  EquationSystem *eqSystem)
  : SolverAlgorithm(realm, part, eqSystem),
    includeDivU_(realm_.get_divU()),
    meshMotion_(realm_.does_mesh_move()),
    velocityRTM_(NULL),
    velocity_(NULL),
    coordinates_(NULL),
    dudx_(NULL),
    density_(NULL),
    viscosity_(NULL),
    massFlowRate_(NULL),
    pecletFunction_(NULL)
{
  // save off data
  stk::mesh::MetaData & meta_data = realm_.meta_data();
  if ( meshMotion_ )
    velocityRTM_ = meta_data.get_field<VectorFieldType>(stk::topology::NODE_RANK, "velocity_rtm");
  else
    velocityRTM_ = meta_data.get_field<VectorFieldType>(stk::topology::NODE_RANK, "velocity");
  velocity_ = meta_data.get_field<VectorFieldType>(stk::topology::NODE_RANK, "velocity");
  coordinates_ = meta_data.get_field<VectorFieldType>(stk::topology::NODE_RANK, realm_.get_coordinates_name());
  dudx_ = meta_data.get_field<GenericFieldType>(stk::topology::NODE_RANK, "dudx");
  density_ = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "density");
  const std::string viscName = realm.is_turbulent()
    ? "effective_viscosity_u" : "viscosity";
  viscosity_ = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, viscName);
  massFlowRate_ = meta_data.get_field<GenericFieldType>(stk::topology::ELEMENT_RANK, "mass_flow_rate_scs");

  // create the peclet blending function
  pecletFunction_ = eqSystem->create_peclet_function(velocity_->name());

  /* Notes:

  Matrix layout is in row major. For a npe = 4 (quad) and nDim = 2:

  RHS = (resUx0, resUy0, resUx1, resUy1, resUx2, resUy2, resUx3, resUy3)

  where Uik = velocity_i_node_k

  The LHS is, therefore,

  row 0: d/dUx0(ResUx0), d/dUy0(ResUx0), ., ., ., .,  d/dUx3(ResUx0), d/dUy3(ResUx0)
  row 1: d/dUx0(ResUy0), d/dUy0(ResUy0), ., ., ., .,  d/dUx3(ResUy0), d/dUy3(ResUy0)

  */
}

//--------------------------------------------------------------------------
//-------- destructor ------------------------------------------------------
//--------------------------------------------------------------------------
AssembleMomentumElemSolverAlgorithm::~AssembleMomentumElemSolverAlgorithm()
{
  delete pecletFunction_;
}

//--------------------------------------------------------------------------
//-------- initialize_connectivity -----------------------------------------
//--------------------------------------------------------------------------
void
AssembleMomentumElemSolverAlgorithm::initialize_connectivity()
{
  eqSystem_->linsys_->buildElemToNodeGraph(partVec_);
}

//--------------------------------------------------------------------------
//-------- execute ---------------------------------------------------------
//--------------------------------------------------------------------------
void
AssembleMomentumElemSolverAlgorithm::execute()
{

  stk::mesh::MetaData & meta_data = realm_.meta_data();

  const int nDim = meta_data.spatial_dimension();

  const bool useShifted = false;

  const double small = 1.0e-16;

  // extract user advection options (allow to potentially change over time)
  const std::string dofName = "velocity";
  const double alpha = realm_.get_alpha_factor(dofName);
  const double alphaUpw = realm_.get_alpha_upw_factor(dofName);
  const double hoUpwind = realm_.get_upw_factor(dofName);
  const bool useLimiter = realm_.primitive_uses_limiter(dofName);
 
  // one minus flavor..
  const double om_alpha = 1.0-alpha;
  const double om_alphaUpw = 1.0-alphaUpw;

  // supplemental algorithm setup
  const size_t supplementalAlgSize = supplementalAlg_.size();
  for ( size_t i = 0; i < supplementalAlgSize; ++i )
    supplementalAlg_[i]->setup();

  // deal with state
  VectorFieldType &velocityNp1 = velocity_->field_of_state(stk::mesh::StateNP1);
  ScalarFieldType &densityNp1 = density_->field_of_state(stk::mesh::StateNP1);

  // define some common selectors
  stk::mesh::Selector s_locally_owned_union = meta_data.locally_owned_part()
    & stk::mesh::selectUnion(partVec_) 
    & !(realm_.get_inactive_selector());

  stk::mesh::BucketVector const& elem_buckets =
    realm_.get_buckets( stk::topology::ELEMENT_RANK, s_locally_owned_union );

  // TODO: Do a parallel_reduce pass to determine the maxes needed.
  const int maxNodesPerElement = 8;
  const int maxNumScsIp = 16;
  const int lhsSize = maxNodesPerElement*nDim*maxNodesPerElement*nDim;
  const int rhsSize = maxNodesPerElement*nDim;

  const int bytes_per_team =
      SharedMemView<double **>::shmem_size(maxNumScsIp, maxNodesPerElement);
  const int bytes_per_thread =
      SharedMemView<double **>::shmem_size(maxNodesPerElement, nDim) +
      SharedMemView<double **>::shmem_size(maxNodesPerElement, nDim) +
      SharedMemView<double **>::shmem_size(maxNodesPerElement, nDim) +
      SharedMemView<double ***>::shmem_size(maxNodesPerElement, nDim, nDim) +
      SharedMemView<double *>::shmem_size(maxNodesPerElement) +
      SharedMemView<double *>::shmem_size(maxNodesPerElement) +
      SharedMemView<double **>::shmem_size(maxNumScsIp, nDim) +
      SharedMemView<double ***>::shmem_size(nDim, maxNumScsIp, maxNodesPerElement) +
      SharedMemView<double ***>::shmem_size(nDim, maxNumScsIp, maxNodesPerElement) +
      SharedMemView<double *>::shmem_size(maxNumScsIp) +
      SharedMemView<double *>::shmem_size(nDim) +
      SharedMemView<double *>::shmem_size(nDim) +
      SharedMemView<double *>::shmem_size(nDim) +
      SharedMemView<double *>::shmem_size(nDim) +
      SharedMemView<double *>::shmem_size(nDim) +
      SharedMemView<double *>::shmem_size(nDim) +
      SharedMemView<double *>::shmem_size(nDim) +
      SharedMemView<double *>::shmem_size(nDim) +
      SharedMemView<double *>::shmem_size(lhsSize) +
      SharedMemView<double *>::shmem_size(rhsSize) +
      SharedMemView<int *>::shmem_size(rhsSize) +
      SharedMemView<stk::mesh::Entity *>::shmem_size(maxNodesPerElement);

  auto team_exec = get_team_policy(elem_buckets.size(), bytes_per_team, bytes_per_thread);

  Kokkos::parallel_for("Nalu::AssembleMomentumElemSolver",
    team_exec, [&] (const DeviceTeam & team) {
    const int ib = team.league_rank();
    const stk::mesh::Bucket & b = *elem_buckets[ib];

    const stk::mesh::Bucket::size_type length   = b.size();

    // extract master element
    MasterElement *meSCS = realm_.get_surface_master_element(b.topology());
    MasterElement *meSCV = realm_.get_volume_master_element(b.topology());

    // extract master element specifics
    const int nodesPerElement = meSCS->nodesPerElement_;
    const int numScsIp = meSCS->numIntPoints_;
    const int *lrscv = meSCS->adjacentNodes();

    SharedMemView<double **> ws_shape_function(team.team_shmem(), numScsIp, nodesPerElement);

    Kokkos::single( Kokkos::PerTeam(team), [&]() {
      // extract shape function
      if ( useShifted )
        meSCS->shifted_shape_fcn(&ws_shape_function(0, 0));
      else
        meSCS->shape_fcn(&ws_shape_function(0, 0));

      // resize possible supplemental element alg
      for ( size_t i = 0; i < supplementalAlgSize; ++i )
        supplementalAlg_[i]->elem_resize(meSCS, meSCV);
    });
    team.team_barrier();

    SharedMemView<double **> ws_velocityNp1;
    SharedMemView<double **> ws_vrtm;
    SharedMemView<double **> ws_coordinates;
    SharedMemView<double **> ws_dudx; // Should be 3D view but there appears to be a bug in the subview assignment from 4D to 3D
    SharedMemView<double *> ws_densityNp1;
    SharedMemView<double *> ws_viscosity;
    SharedMemView<double **> ws_scs_areav;
    SharedMemView<double **> ws_dndx; // Should be 3D view but there appears to be a bug in the subview assignment from 4D to 3D
    SharedMemView<double **> ws_deriv; // Should be 3D view but there appears to be a bug in the subview assignment from 4D to 3D
    SharedMemView<double *> ws_det_j;

    // ip values
    SharedMemView<double *> uIp;
    // extrapolated value from the L/R direction
    SharedMemView<double *> uIpL;
    SharedMemView<double *> uIpR;
    // limiter values from the L/R direction, 0:1
    SharedMemView<double *> limitL;
    SharedMemView<double *> limitR;
    // extrapolated gradient from L/R direction
    SharedMemView<double *> duL;
    SharedMemView<double *> duR;

    SharedMemView<double *> coordIp;

    SharedMemView<double *> lhs;
    SharedMemView<double *> rhs;
    SharedMemView<stk::mesh::Entity *> connected_nodes;
    SharedMemView<int *> localIdsScratch;
    {
      ws_velocityNp1 = Kokkos::subview(
          SharedMemView<double ***> (team.team_shmem(), team.team_size(), nodesPerElement, nDim),
          team.team_rank(), Kokkos::ALL(), Kokkos::ALL());
      ws_vrtm = Kokkos::subview(
          SharedMemView<double ***> (team.team_shmem(), team.team_size(), nodesPerElement, nDim),
          team.team_rank(), Kokkos::ALL(), Kokkos::ALL());
      ws_coordinates = Kokkos::subview(
          SharedMemView<double ***> (team.team_shmem(), team.team_size(), nodesPerElement, nDim),
          team.team_rank(), Kokkos::ALL(), Kokkos::ALL());
      ws_dudx = Kokkos::subview(
          SharedMemView<double ***> (team.team_shmem(), team.team_size(), nodesPerElement, nDim * nDim),
          team.team_rank(), Kokkos::ALL(), Kokkos::ALL());
      ws_densityNp1 = Kokkos::subview(
          SharedMemView<double **> (team.team_shmem(), team.team_size(), nodesPerElement),
          team.team_rank(), Kokkos::ALL());
      ws_viscosity = Kokkos::subview(
          SharedMemView<double **> (team.team_shmem(), team.team_size(), nodesPerElement),
          team.team_rank(), Kokkos::ALL());
      ws_scs_areav = Kokkos::subview(
          SharedMemView<double ***> (team.team_shmem(), team.team_size(), numScsIp, nDim),
          team.team_rank(), Kokkos::ALL(), Kokkos::ALL());
      ws_dndx = Kokkos::subview(
          SharedMemView<double ***> (team.team_shmem(), team.team_size(), numScsIp, nodesPerElement * nDim),
          team.team_rank(), Kokkos::ALL(), Kokkos::ALL());
      ws_deriv = Kokkos::subview(
          SharedMemView<double ***> (team.team_shmem(), team.team_size(), numScsIp, nodesPerElement * nDim),
          team.team_rank(), Kokkos::ALL(), Kokkos::ALL());
      ws_det_j = Kokkos::subview(
          SharedMemView<double **> (team.team_shmem(), team.team_size(), numScsIp),
          team.team_rank(), Kokkos::ALL());

      // ip values
      uIp = Kokkos::subview(
          SharedMemView<double **> (team.team_shmem(), team.team_size(), nDim),
          team.team_rank(), Kokkos::ALL());
      // extrapolated value from the L/R direction
      uIpL = Kokkos::subview(
          SharedMemView<double **> (team.team_shmem(), team.team_size(), nDim),
          team.team_rank(), Kokkos::ALL());
      uIpR = Kokkos::subview(
          SharedMemView<double **> (team.team_shmem(), team.team_size(), nDim),
          team.team_rank(), Kokkos::ALL());
      // limiter values from the L/R direction, 0:1
      limitL = Kokkos::subview(
          SharedMemView<double **> (team.team_shmem(), team.team_size(), nDim),
          team.team_rank(), Kokkos::ALL());
      limitR = Kokkos::subview(
          SharedMemView<double **> (team.team_shmem(), team.team_size(), nDim),
          team.team_rank(), Kokkos::ALL());
      // extrapolated gradient from L/R direction
      duL = Kokkos::subview(
          SharedMemView<double **> (team.team_shmem(), team.team_size(), nDim),
          team.team_rank(), Kokkos::ALL());
      duR = Kokkos::subview(
          SharedMemView<double **> (team.team_shmem(), team.team_size(), nDim),
          team.team_rank(), Kokkos::ALL());

      coordIp = Kokkos::subview(
          SharedMemView<double **> (team.team_shmem(), team.team_size(), nDim),
          team.team_rank(), Kokkos::ALL());

      lhs = Kokkos::subview(
          SharedMemView<double **> (team.team_shmem(), team.team_size(), lhsSize),
          team.team_rank(), Kokkos::ALL());
      rhs = Kokkos::subview(
          SharedMemView<double **> (team.team_shmem(), team.team_size(), rhsSize),
          team.team_rank(), Kokkos::ALL());
      connected_nodes = Kokkos::subview(
          SharedMemView<stk::mesh::Entity **> (team.team_shmem(), team.team_size(), nodesPerElement),
          team.team_rank(), Kokkos::ALL());
      localIdsScratch = Kokkos::subview(
          SharedMemView<int **> (team.team_shmem(), team.team_size(), rhsSize),
          team.team_rank(), Kokkos::ALL());
    }

    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, length), [&] (const size_t k) {
      // get elem
      const stk::mesh::Entity elem = b[k];

      // zero lhs/rhs
      for ( int p = 0; p < lhsSize; ++p )
        lhs[p] = 0.0;
      for ( int p = 0; p < rhsSize; ++p )
        rhs[p] = 0.0;

      for ( int d = 0; d < nDim; ++d ) {
        limitL(d) = 1.;
        limitR(d) = 1.;
      }

      // ip data for this element; scs and scv
      const double *mdot = stk::mesh::field_data(*massFlowRate_, b, k );

      //===============================================
      // gather nodal data; this is how we do it now..
      //===============================================
      stk::mesh::Entity const * node_rels = b.begin_nodes(k);
      int num_nodes = b.num_nodes(k);

      // sanity check on num nodes
      //ThrowAssert( num_nodes == nodesPerElement );

      for ( int ni = 0; ni < num_nodes; ++ni ) {
        stk::mesh::Entity node = node_rels[ni];

        // set connected nodes
        connected_nodes[ni] = node;

        // pointers to real data
        const double * uNp1   =  stk::mesh::field_data(velocityNp1, node);
        const double * vrtm   = stk::mesh::field_data(*velocityRTM_, node);
        const double * coords =  stk::mesh::field_data(*coordinates_, node);
        const double * du     =  stk::mesh::field_data(*dudx_, node);
        const double rhoNp1   = *stk::mesh::field_data(densityNp1, node);
        const double mu       = *stk::mesh::field_data(*viscosity_, node);

        // gather scalars
        ws_densityNp1[ni] = rhoNp1;
        ws_viscosity[ni] = mu;

        for ( int i=0; i < nDim; ++i ) {
          ws_velocityNp1(ni, i) = uNp1[i];
          ws_vrtm(ni, i) = vrtm[i];
          ws_coordinates(ni, i) = coords[i];
          // gather tensor
          const int row_dudx = i*nDim;
          for ( int j=0; j < nDim; ++j ) {
            ws_dudx(ni, i*nDim + j) = du[row_dudx+j];
          }
        }
      }

      // compute geometry
      double scs_error = 0.0;
      meSCS->determinant(1, &ws_coordinates(0, 0), &ws_scs_areav(0, 0), &scs_error);

      // compute dndx
      meSCS->grad_op(1, &ws_coordinates(0, 0), &ws_dndx(0, 0), &ws_deriv(0, 0),
          &ws_det_j(0), &scs_error);

      for ( int ip = 0; ip < numScsIp; ++ip ) {

        const int ipNdim = ip*nDim;

        const int offSetSF = ip*nodesPerElement;

        // left and right nodes for this ip
        const int il = lrscv[2*ip];
        const int ir = lrscv[2*ip+1];

        // save off mdot
        const double tmdot = mdot[ip];

        // save off some offsets
        const int ilNdim = il*nDim;
        const int irNdim = ir*nDim;

        // zero out values of interest for this ip
        for ( int j = 0; j < nDim; ++j ) {
          uIp[j] = 0.0;
          coordIp[j] = 0.0;
        }

        // compute scs point values; offset to Shape Function; sneak in divU
        double muIp = 0.0;
        double divU = 0.0;
        for ( int ic = 0; ic < nodesPerElement; ++ic ) {
          const double r = ws_shape_function(ip, ic);
          muIp += r*ws_viscosity[ic];
          for ( int j = 0; j < nDim; ++j ) {
            coordIp[j] += r*ws_coordinates(ic, j);
            const double uj = ws_velocityNp1(ic, j);
            uIp[j] += r*uj;
            divU += uj*ws_dndx(ip, ic*nDim + j);
          }
        }

        // udotx; left and right extrapolation
        double udotx = 0.0;
        for (int i = 0; i < nDim; ++i ) {
          // udotx
          const double dxi = ws_coordinates(ir, i)-ws_coordinates(il, i);
          const double ui = 0.5*(ws_vrtm(il, i) + ws_vrtm(ir, i));
          udotx += ui*dxi;
          // extrapolation du
          duL[i] = 0.0;
          duR[i] = 0.0;
          for(int j = 0; j < nDim; ++j ) {
            const double dxjL = coordIp[j] - ws_coordinates(il, j);
            const double dxjR = ws_coordinates(ir, j) - coordIp[j];
            duL[i] += dxjL*ws_dudx(il, i*nDim + j);
            duR[i] += dxjR*ws_dudx(ir, i*nDim + j);
          }
        }

        // Peclet factor; along the edge is fine
        const double diffIp = 0.5*(ws_viscosity[il]/ws_densityNp1[il]
                                   + ws_viscosity[ir]/ws_densityNp1[ir]);
        const double pecfac = pecletFunction_->execute(std::abs(udotx)/(diffIp+small));
        const double om_pecfac = 1.0-pecfac;
	
        // determine limiter if applicable
        if ( useLimiter ) {
          for ( int i = 0; i < nDim; ++i ) {
            const double dq = ws_velocityNp1(ir, i) - ws_velocityNp1(il, i);
            const double dqMl = 2.0*2.0*duL[i] - dq;
            const double dqMr = 2.0*2.0*duR[i] - dq;
            limitL[i] = van_leer(dqMl, dq, small);
            limitR[i] = van_leer(dqMr, dq, small);
          }
        }
	
        // final upwind extrapolation; with limiter
        for ( int i = 0; i < nDim; ++i ) {
          uIpL[i] = ws_velocityNp1(il, i) + duL[i]*hoUpwind*limitL[i];
          uIpR[i] = ws_velocityNp1(ir, i) - duR[i]*hoUpwind*limitR[i];
        }

        // assemble advection; rhs and upwind contributions; add in divU stress (explicit)
        for ( int i = 0; i < nDim; ++i ) {

          // 2nd order central
          const double uiIp = uIp[i];

          // upwind
          const double uiUpwind = (tmdot > 0) ? alphaUpw*uIpL[i] + (om_alphaUpw)*uiIp
            : alphaUpw*uIpR[i] + (om_alphaUpw)*uiIp;

          // generalized central (2nd and 4th order)
          const double uiHatL = alpha*uIpL[i] + om_alpha*uiIp;
          const double uiHatR = alpha*uIpR[i] + om_alpha*uiIp;
          const double uiCds = 0.5*(uiHatL + uiHatR);

          // total advection; pressure contribution in time term
          const double aflux = tmdot*(pecfac*uiUpwind + om_pecfac*uiCds);

          // divU stress term
          const double divUstress = 2.0/3.0*muIp*divU*ws_scs_areav(ip, i)*includeDivU_;

          const int indexL = ilNdim + i;
          const int indexR = irNdim + i;

          const int rowL = indexL*nodesPerElement*nDim;
          const int rowR = indexR*nodesPerElement*nDim;

          const int rLiL_i = rowL+ilNdim+i;
          const int rLiR_i = rowL+irNdim+i;
          const int rRiR_i = rowR+irNdim+i;
          const int rRiL_i = rowR+ilNdim+i;

          // right hand side; L and R
          rhs[indexL] -= aflux + divUstress;
          rhs[indexR] += aflux + divUstress;

          // advection operator sens; all but central

          // upwind advection (includes 4th); left node
          const double alhsfacL = 0.5*(tmdot+std::abs(tmdot))*pecfac*alphaUpw
            + 0.5*alpha*om_pecfac*tmdot;
          lhs[rLiL_i] += alhsfacL;
          lhs[rRiL_i] -= alhsfacL;

          // upwind advection (includes 4th); right node
          const double alhsfacR = 0.5*(tmdot-std::abs(tmdot))*pecfac*alphaUpw
            + 0.5*alpha*om_pecfac*tmdot;
          lhs[rRiR_i] -= alhsfacR;
          lhs[rLiR_i] += alhsfacR;

        }

        for ( int ic = 0; ic < nodesPerElement; ++ic ) {

          const int icNdim = ic*nDim;

          // shape function
          const double r = ws_shape_function(ip, ic);

          // advection and diffison

          // upwind (il/ir) handled above; collect terms on alpha and alphaUpw
          const double lhsfacAdv = r*tmdot*(pecfac*om_alphaUpw + om_pecfac*om_alpha);

          for ( int i = 0; i < nDim; ++i ) {

            const int indexL = ilNdim + i;
            const int indexR = irNdim + i;

            const int rowL = indexL*nodesPerElement*nDim;
            const int rowR = indexR*nodesPerElement*nDim;

            const int rLiC_i = rowL+icNdim+i;
            const int rRiC_i = rowR+icNdim+i;

            // advection operator  lhs; rhs handled above
            // lhs; il then ir
            lhs[rLiC_i] += lhsfacAdv;
            lhs[rRiC_i] -= lhsfacAdv;

            // viscous stress
            const int offSetDnDx = nDim*nodesPerElement*ip + icNdim;
            double lhs_riC_i = 0.0;
            for ( int j = 0; j < nDim; ++j ) {

              const double axj = ws_scs_areav(ip, j);
              const double uj = ws_velocityNp1(ic, j);

              // -mu*dui/dxj*A_j; fixed i over j loop; see below..
              const double lhsfacDiff_i = -muIp*ws_dndx(ip, ic*nDim + j)*axj;
              // lhs; il then ir
              lhs_riC_i += lhsfacDiff_i;

              // -mu*duj/dxi*A_j
              const double lhsfacDiff_j = -muIp*ws_dndx(ip, ic*nDim + i)*axj;
              // lhs; il then ir
              lhs[rowL+icNdim+j] += lhsfacDiff_j;
              lhs[rowR+icNdim+j] -= lhsfacDiff_j;
              // rhs; il then ir
              rhs[indexL] -= lhsfacDiff_j*uj;
              rhs[indexR] += lhsfacDiff_j*uj;
            }

            // deal with accumulated lhs and flux for -mu*dui/dxj*Aj
            lhs[rLiC_i] += lhs_riC_i;
            lhs[rRiC_i] -= lhs_riC_i;
            const double ui = ws_velocityNp1(ic, i);
            rhs[indexL] -= lhs_riC_i*ui;
            rhs[indexR] += lhs_riC_i*ui;

          }
        }
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
AssembleMomentumElemSolverAlgorithm::van_leer(
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
