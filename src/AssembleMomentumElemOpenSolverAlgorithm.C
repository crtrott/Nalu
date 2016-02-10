/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


// nalu
#include <AssembleMomentumElemOpenSolverAlgorithm.h>
#include <EquationSystem.h>
#include <FieldTypeDef.h>
#include <LinearSystem.h>
#include <PecletFunction.h>
#include <Realm.h>
#include <SolutionOptions.h>
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
// AssembleMomentumElemOpenSolverAlgorithm - lhs for momentum open bc
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
AssembleMomentumElemOpenSolverAlgorithm::AssembleMomentumElemOpenSolverAlgorithm(
  Realm &realm,
  stk::mesh::Part *part,
  EquationSystem *eqSystem)
  : SolverAlgorithm(realm, part, eqSystem),
    includeDivU_(realm_.get_divU()),
    pecletFunction_(NULL)
{
  // save off fields
  stk::mesh::MetaData & meta_data = realm_.meta_data();
  velocity_ = meta_data.get_field<VectorFieldType>(stk::topology::NODE_RANK, "velocity");
  dudx_ = meta_data.get_field<GenericFieldType>(stk::topology::NODE_RANK, "dudx");
  coordinates_ = meta_data.get_field<VectorFieldType>(stk::topology::NODE_RANK, realm_.get_coordinates_name());
  density_ = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "density");
  const std::string viscName = realm.is_turbulent()
    ? "effective_viscosity_u" : "viscosity";
  viscosity_ = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, viscName);
  exposedAreaVec_ = meta_data.get_field<GenericFieldType>(meta_data.side_rank(), "exposed_area_vector");
  openMassFlowRate_ = meta_data.get_field<GenericFieldType>(meta_data.side_rank(), "open_mass_flow_rate");
  velocityBc_ = meta_data.get_field<VectorFieldType>(stk::topology::NODE_RANK, "open_velocity_bc");

  // create the peclet blending function
  pecletFunction_ = eqSystem->create_peclet_function(velocity_->name());
}

//--------------------------------------------------------------------------
//-------- initialize_connectivity -----------------------------------------
//--------------------------------------------------------------------------
void
AssembleMomentumElemOpenSolverAlgorithm::initialize_connectivity()
{
  eqSystem_->linsys_->buildFaceElemToNodeGraph(partVec_);
}

//--------------------------------------------------------------------------
//-------- destructor ------------------------------------------------------
//--------------------------------------------------------------------------
AssembleMomentumElemOpenSolverAlgorithm::~AssembleMomentumElemOpenSolverAlgorithm()
{
  delete pecletFunction_;
}

//--------------------------------------------------------------------------
//-------- execute ---------------------------------------------------------
//--------------------------------------------------------------------------
void
AssembleMomentumElemOpenSolverAlgorithm::execute()
{
  stk::mesh::BulkData & bulk_data = realm_.bulk_data();
  stk::mesh::MetaData & meta_data = realm_.meta_data();

  const int nDim = meta_data.spatial_dimension();

  const double small = 1.0e-16;

  // extract user advection options (allow to potentially change over time)
  const std::string dofName = "velocity";
  const double alphaUpw = realm_.get_alpha_upw_factor(dofName);
  const double hoUpwind = realm_.get_upw_factor(dofName);
  
  // one minus flavor..
  const double om_alphaUpw = 1.0-alphaUpw;

  // nearest face entrainment
  const double nfEntrain = realm_.solutionOptions_->nearestFaceEntrain_;
  const double om_nfEntrain = 1.0-nfEntrain;

  // deal with state
  VectorFieldType &velocityNp1_field = velocity_->field_of_state(stk::mesh::StateNP1);
  ScalarFieldType &densityNp1_field = density_->field_of_state(stk::mesh::StateNP1);

  // define some common selectors
  stk::mesh::Selector s_locally_owned_union = meta_data.locally_owned_part()
    &stk::mesh::selectUnion(partVec_);

  stk::mesh::BucketVector const& face_buckets =
    realm_.get_buckets( meta_data.side_rank(), s_locally_owned_union );

  const int maxNodesPerElement = 8;
  const int maxNumScsIp = 16;
  const int maxNodesPerFace = 4;
  const int maxNumScsBip = 8;

  const int maxlhsSize = maxNodesPerElement*nDim*maxNodesPerElement*nDim;
  const int maxrhsSize = maxNodesPerElement*nDim;

  const int bytes_per_team =
      SharedMemView<double **>::shmem_size(maxNumScsBip, maxNodesPerFace) +
      SharedMemView<double **>::shmem_size(maxNumScsIp, maxNodesPerElement);

  const int bytes_per_thread =
      SharedMemView<double *>::shmem_size(maxlhsSize) +
      SharedMemView<double *>::shmem_size(maxrhsSize) +
      SharedMemView<int *>::shmem_size(maxrhsSize) +
      SharedMemView<int *>::shmem_size(maxNodesPerFace) +
      SharedMemView<stk::mesh::Entity *>::shmem_size(maxNodesPerElement) +
      SharedMemView<double **>::shmem_size(maxNodesPerElement, nDim) +
      SharedMemView<double ***>::shmem_size(maxNodesPerFace, nDim, nDim) +
      SharedMemView<double **>::shmem_size(maxNodesPerElement, nDim) +
      SharedMemView<double *>::shmem_size(maxNodesPerFace) +
      SharedMemView<double **>::shmem_size(maxNodesPerFace, nDim) +
      SharedMemView<double ***>::shmem_size(maxNumScsBip, maxNodesPerElement, nDim) +
      SharedMemView<double *>::shmem_size(maxNumScsBip) +
      SharedMemView<double *>::shmem_size(nDim) +
      SharedMemView<double *>::shmem_size(nDim) +
      SharedMemView<double *>::shmem_size(nDim) +
      SharedMemView<double *>::shmem_size(nDim) +
      SharedMemView<double *>::shmem_size(nDim) +
      SharedMemView<double *>::shmem_size(nDim);

  auto team_exec = get_team_policy(face_buckets.size(), bytes_per_team, bytes_per_thread);

  Kokkos::parallel_for("Nalu::AssembleMomentumElemOpenSolver",
      team_exec, [&] (const DeviceTeam & team) {
      const stk::mesh::Bucket & b = *face_buckets[team.league_rank()];

      // extract connected element topology
      const auto first_elem = bulk_data.begin_elements(b[0])[0];
      stk::topology theElemTopo = bulk_data.bucket(first_elem).topology();

      // volume master element
      MasterElement *meSCS = realm_.get_surface_master_element(theElemTopo);
      const int nodesPerElement = meSCS->nodesPerElement_;
      const int numScsIp = meSCS->numIntPoints_;

      // face master element
      MasterElement *meFC = realm_.get_surface_master_element(b.topology());
      const int nodesPerFace = meFC->nodesPerElement_;
      const int numScsBip = meFC->numIntPoints_;

      // resize some things; matrix related
      const int lhsSize = nodesPerElement*nDim*nodesPerElement*nDim;
      const int rhsSize = nodesPerElement*nDim;
      const int scratch_level = 2;
      SharedMemView<int *> face_node_ordinal_vec(team.thread_scratch(scratch_level), nodesPerFace);

      SharedMemView<double *> lhs(team.thread_scratch(scratch_level), lhsSize);
      SharedMemView<double *> rhs(team.thread_scratch(scratch_level), rhsSize);
      SharedMemView<int *> localIdsScratch(team.thread_scratch(scratch_level), rhsSize);
      SharedMemView<stk::mesh::Entity *> connected_nodes(team.thread_scratch(scratch_level), nodesPerElement);

      // algorithm related; element
      SharedMemView<double **> velocityNp1(team.thread_scratch(scratch_level), nodesPerElement, nDim);
      SharedMemView<double ***> dudx(team.thread_scratch(scratch_level), nodesPerFace, nDim, nDim);
      SharedMemView<double **> coordinates(team.thread_scratch(scratch_level), nodesPerElement, nDim);
      SharedMemView<double *> viscosity(team.thread_scratch(scratch_level), nodesPerFace);
      SharedMemView<double **> bcVelocity(team.thread_scratch(scratch_level), nodesPerFace, nDim);
      SharedMemView<double ***> dndx(team.thread_scratch(scratch_level), numScsBip, nodesPerElement, nDim);
      SharedMemView<double *> det_j(team.thread_scratch(scratch_level), numScsBip);

      SharedMemView<double *> uBip(team.thread_scratch(scratch_level), nDim);
      SharedMemView<double *> uScs(team.thread_scratch(scratch_level), nDim);
      SharedMemView<double *> uBipExtrap(team.thread_scratch(scratch_level), nDim);
      SharedMemView<double *> uspecBip(team.thread_scratch(scratch_level), nDim);
      SharedMemView<double *> coordBip(team.thread_scratch(scratch_level), nDim);
      SharedMemView<double *> nx(team.thread_scratch(scratch_level), nDim);

      SharedMemView<double **> face_shape_function(team.team_scratch(scratch_level), numScsBip, nodesPerFace);
      SharedMemView<double **> shape_function(team.team_scratch(scratch_level), numScsIp, nodesPerElement);
      // shape function
      Kokkos::single(Kokkos::PerTeam(team), [&]() {
        meSCS->shape_fcn(&shape_function(0, 0));
        meFC->shape_fcn(&face_shape_function(0, 0));
      });
      team.team_barrier();

      const auto length = b.size();

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
          viscosity[ni] = *stk::mesh::field_data(*viscosity_, node);

          // gather vectors
          double * uspec = stk::mesh::field_data(*velocityBc_, node);
          double * Gjui = stk::mesh::field_data(*dudx_, node);
          const int niNdim = ni*nDim;
          const int row_p_dudx = niNdim*nDim;
          for ( int i=0; i < nDim; ++i ) {
            bcVelocity(ni, i) = uspec[i];
            // gather tensor
            const int row_dudx = i*nDim;
            for ( int j = 0; j < nDim; ++j ) {
              dudx(ni, i, j) = Gjui[row_dudx + j];
            }
          }
        }

        // pointer to face data
        const double * areaVec = stk::mesh::field_data(*exposedAreaVec_, face);

        // extract the connected element to this exposed face; should be single in size!
        stk::mesh::Entity const * face_elem_rels = bulk_data.begin_elements(face);
        //ThrowAssert( bulk_data.num_elements(face) == 1 );

        // get element; its face ordinal number and populate face_node_ordinal_vec
        stk::mesh::Entity element = face_elem_rels[0];
        const stk::mesh::ConnectivityOrdinal* face_elem_ords = bulk_data.begin_element_ordinals(face);
        const int face_ordinal = face_elem_ords[0];
        theElemTopo.side_node_ordinals(face_ordinal, &face_node_ordinal_vec[0]);

        // mapping from ip to nodes for this ordinal
        const int *ipNodeMap = meSCS->ipNodeMap(face_ordinal);
        const int *faceIpNodeMap = meFC->ipNodeMap();

        //==========================================
        // gather nodal data off of element
        //==========================================
        stk::mesh::Entity const * elem_node_rels = bulk_data.begin_nodes(element);
        int num_nodes = bulk_data.num_nodes(element);
        // sanity check on num nodes
        ThrowAssert( num_nodes == nodesPerElement );
        for ( int ni = 0; ni < num_nodes; ++ni ) {
          stk::mesh::Entity node = elem_node_rels[ni];
          // set connected nodes
          connected_nodes[ni] = node;
          // gather vectors
          double * uNp1 = stk::mesh::field_data(velocityNp1_field, node);
          double * coords = stk::mesh::field_data(*coordinates_, node);
          const int offSet = ni*nDim;
          for ( int j=0; j < nDim; ++j ) {
            velocityNp1(ni, j) = uNp1[j];
            coordinates(ni, j) = coords[j];
          }
        }

        // compute dndx
        double scs_error = 0.0;
        meSCS->face_grad_op(1, face_ordinal, &coordinates(0, 0), &dndx(0, 0, 0), &det_j[0], &scs_error);

        // loop over boundary ips
        for ( int ip = 0; ip < numScsBip; ++ip ) {

          const int opposingNode = meSCS->opposingNodes(face_ordinal,ip);
          const int nearestNode = ipNodeMap[ip];
          const int opposingScsIp = meSCS->opposingFace(face_ordinal,ip);
          const int localFaceNode = faceIpNodeMap[ip];

          // offset for bip area vector and types of shape function
          const int faceOffSet = ip*nDim;
          const int offSetSF_face = ip*nodesPerFace;
          const int offSetSF_elem = opposingScsIp*nodesPerElement;

          // left and right nodes; right is on the face; left is the opposing node
          stk::mesh::Entity nodeL = elem_node_rels[opposingNode];
          stk::mesh::Entity nodeR = elem_node_rels[nearestNode];

          // zero out vector quantities
          double asq = 0.0;
          for ( int j = 0; j < nDim; ++j ) {
            uBip[j] = 0.0;
            uScs[j] = 0.0;
            uspecBip[j] = 0.0;
            coordBip[j] = 0.0;
            const double axj = areaVec[faceOffSet+j];
            asq += axj*axj;
          }
          const double amag = std::sqrt(asq);

          // interpolate to bip
          double viscBip = 0.0;
          for ( int ic = 0; ic < nodesPerFace; ++ic ) {
            const double r = face_shape_function(ip, ic);
            viscBip += r*viscosity[ic];
            const int nn = face_node_ordinal_vec[ic];
            for ( int j = 0; j < nDim; ++j ) {
              uspecBip[j] += r*bcVelocity(ic, j);
              uBip[j] += r*velocityNp1(nn, j);
              coordBip[j] += r*coordinates(nn, j);
            }
          }

          // data at interior opposing face
          for ( int ic = 0; ic < nodesPerElement; ++ic ) {
            const double r = shape_function(opposingScsIp, ic);
            for ( int j = 0; j < nDim; ++j ) {
              uScs[j] += r*velocityNp1(ic, j);
            }
          }

          // Peclet factor; along the edge is fine
          const double densL   = *stk::mesh::field_data(densityNp1_field, nodeL);
          const double densR   = *stk::mesh::field_data(densityNp1_field, nodeR);
          const double viscL   = *stk::mesh::field_data(*viscosity_, nodeL);
          const double viscR   = *stk::mesh::field_data(*viscosity_, nodeR);
          const double *uNp1L  =  stk::mesh::field_data(velocityNp1_field, nodeL);
          const double *uNp1R  =  stk::mesh::field_data(velocityNp1_field, nodeR);
          const double *coordL =  stk::mesh::field_data(*coordinates_, nodeL);
          const double *coordR =  stk::mesh::field_data(*coordinates_, nodeR);

          double udotx = 0.0;
          const int row_p_dudxR = localFaceNode*nDim*nDim; // tricky here with localFaceNode
          for ( int i = 0; i < nDim; ++i ) {
            const double dxi = coordR[i]  - coordL[i];
            udotx += 0.5*dxi*(uNp1L[i] + uNp1R[i]);
            nx[i] = areaVec[faceOffSet+i]/amag;
            // extrapolation
            double duR = 0.0;
            for ( int j = 0; j < nDim; ++j ) {
              double dxj = coordBip[j] - coordR[j];
              duR += dxj*dudx(localFaceNode, i, j)*hoUpwind;
            }
            uBipExtrap[i] = uNp1R[i] + duR;
          }

          const double diffIp = 0.5*(viscL/densL + viscR/densR);
          const double pecfac = pecletFunction_->execute(std::abs(udotx)/(diffIp+small));
          const double om_pecfac = 1.0-pecfac;

          //================================
          // advection first
          //================================
          const double tmdot = mdot[ip];

          // advection; leaving the domain
          if ( tmdot > 0.0 ) {

            for ( int i = 0; i < nDim; ++i ) {

              const int indexR = nearestNode*nDim + i;
              const int rowR = indexR*nodesPerElement*nDim;

              // central
              const double uiIp = uBip[i];

              // upwind
              const double uiUpwind = alphaUpw*uBipExtrap[i] + om_alphaUpw*uiIp;

              // total advection; pressure contribution in time expression
              const double aflux = tmdot*(pecfac*uiUpwind+om_pecfac*uiIp);

              rhs[indexR] -= aflux;

              // upwind lhs
              lhs[rowR+indexR] += tmdot*pecfac*alphaUpw;

              // central part
              const double fac = tmdot*(pecfac*om_alphaUpw+om_pecfac);
              for ( int ic = 0; ic < nodesPerFace; ++ic ) {
                const double r = face_shape_function(ip, ic);
                const int nn = face_node_ordinal_vec[ic];
                lhs[rowR+nn*nDim+i] += r*fac;
              }
            }
          }
          else {
            // extrainment
            double ubipnx = 0.0;
            double ubipExtrapnx = 0.0;
            double uscsnx = 0.0;
            double uspecbipnx = 0.0;
            for ( int j = 0; j < nDim; ++j ) {
              const double nj = nx[j];
              ubipnx += uBip[j]*nj;
              ubipExtrapnx += uBipExtrap[j]*nj;
              uscsnx += uScs[j]*nj;
              uspecbipnx += uspecBip[j]*nj;
            }

            for ( int i = 0; i < nDim; ++i ) {
              const int indexR = nearestNode*nDim + i;
              const int rowR = indexR*nodesPerElement*nDim;
              const double nxi = nx[i];

              // total advection; with tangeant entrain
              const double aflux = tmdot*(pecfac*ubipExtrapnx+om_pecfac*
                                          (nfEntrain*ubipnx + om_nfEntrain*uscsnx))*nxi
                + tmdot*(uspecBip[i] - uspecbipnx*nxi);

              rhs[indexR] -= aflux;

              // upwind and central
              for ( int j = 0; j < nDim; ++j ) {
                const double nxinxj = nxi*nx[j];

                // upwind
                lhs[rowR+nearestNode*nDim+j] += tmdot*pecfac*alphaUpw*nxinxj;

                // central part; exposed face
                double fac = tmdot*(pecfac*om_alphaUpw+om_pecfac*nfEntrain)*nxinxj;
                for ( int ic = 0; ic < nodesPerFace; ++ic ) {
                  const double r = face_shape_function(ip, ic);
                  const int nn = face_node_ordinal_vec[ic];
                  lhs[rowR+nn*nDim+j] += r*fac;
                }

                // central part; scs face
                fac = tmdot*om_pecfac*om_nfEntrain*nxinxj;
                for ( int ic = 0; ic < nodesPerElement; ++ic ) {
                  const double r = shape_function(opposingScsIp, ic);
                  lhs[rowR+ic*nDim+j] += r*fac;
                }

              }
            }
          }

          //================================
          // diffusion second
          //================================
          for ( int ic = 0; ic < nodesPerElement; ++ic ) {

            const int offSetDnDx = nDim*nodesPerElement*ip + ic*nDim;

            for ( int j = 0; j < nDim; ++j ) {

              const double axj = areaVec[faceOffSet+j];
              const double dndxj = dndx(ip, ic, j);
              const double uxj = velocityNp1(ic, j);

              const double divUstress = 2.0/3.0*viscBip*dndxj*uxj*axj*includeDivU_;

              for ( int i = 0; i < nDim; ++i ) {
                // matrix entries
                int indexR = nearestNode*nDim + i;
                int rowR = indexR*nodesPerElement*nDim;

                const double dndxi = dndx(ip, ic, i);
                const double uxi = velocityNp1(ic, i);
                const double nxi = nx[i];
                const double om_nxinxi = 1.0-nxi*nxi;

                // -mu*dui/dxj*Aj(1.0-nini); sneak in divU (explicit)
                double lhsfac = -viscBip*dndxj*axj*om_nxinxi;
                lhs[rowR+ic*nDim+i] += lhsfac;
                rhs[indexR] -= lhsfac*uxi + divUstress*om_nxinxi;

                // -mu*duj/dxi*Aj(1.0-nini)
                lhsfac = -viscBip*dndxi*axj*om_nxinxi;
                lhs[rowR+ic*nDim+j] += lhsfac;
                rhs[indexR] -= lhsfac*uxj;

                // now we need the -nx*ny*Fy - nx*nz*Fz part
                for ( int l = 0; l < nDim; ++l ) {

                  if ( i != l ) {
                    const double nxinxl = nxi*nx[l];
                    const double uxl = velocityNp1(ic, l);
                    const double dndxl = dndx(ip, ic, l);

                    // +ni*nl*mu*dul/dxj*Aj; sneak in divU (explicit)
                    lhsfac = viscBip*dndxj*axj*nxinxl;
                    lhs[rowR+ic*nDim+l] += lhsfac;
                    rhs[indexR] -= lhsfac*uxl + divUstress*nxinxl;

                    // +ni*nl*mu*duj/dxl*Aj
                    lhsfac = viscBip*dndxl*axj*nxinxl;
                    lhs[rowR+ic*nDim+j] += lhsfac;
                    rhs[indexR] -= lhsfac*uxj;
                  }
                }
              }
            }
          }
        }

        apply_coeff(connected_nodes, rhs, lhs, localIdsScratch, __FILE__);

      });
    });
}

} // namespace nalu
} // namespace Sierra
