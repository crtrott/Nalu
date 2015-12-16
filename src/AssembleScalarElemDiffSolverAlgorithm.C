/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


// nalu
#include <AssembleScalarElemDiffSolverAlgorithm.h>
#include <EquationSystem.h>
#include <SolverAlgorithm.h>
#include <ScalarElemDiffusionFunctor.h>

#include <FieldTypeDef.h>
#include <LinearSystem.h>
#include <Realm.h>
#include <SupplementalAlgorithm.h>
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
// AssembleScalarElemDiffSolverAlgorithm - add LHS/RHS for scalar diff
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
AssembleScalarElemDiffSolverAlgorithm::AssembleScalarElemDiffSolverAlgorithm(
  Realm &realm,
  stk::mesh::Part *part,
  EquationSystem *eqSystem,
  ScalarFieldType *scalarQ,
  VectorFieldType *dqdx,
  ScalarFieldType *diffFluxCoeff,
  bool useCollocation)
  : SolverAlgorithm(realm, part, eqSystem),
    scalarQ_(scalarQ),
    diffFluxCoeff_(diffFluxCoeff),
    useCollocation_(useCollocation)
{

  // save off fields
  stk::mesh::MetaData & meta_data = realm_.meta_data();
  coordinates_ = meta_data.get_field<VectorFieldType>(stk::topology::NODE_RANK, realm_.get_coordinates_name());
}

//--------------------------------------------------------------------------
//-------- initialize_connectivity -----------------------------------------
//--------------------------------------------------------------------------
void
AssembleScalarElemDiffSolverAlgorithm::initialize_connectivity()
{
  eqSystem_->linsys_->buildElemToNodeGraph(partVec_);
}

//--------------------------------------------------------------------------
//-------- execute ---------------------------------------------------------
//--------------------------------------------------------------------------
void
AssembleScalarElemDiffSolverAlgorithm::execute()
{

  stk::mesh::BulkData & bulk_data = realm_.bulk_data();
  stk::mesh::MetaData & meta_data = realm_.meta_data();

  const int nDim = meta_data.spatial_dimension();

  const int maxElementsPerBucket = 512;
  const int maxNodesPerElement = 8;
  const int maxNumScsIp = 16;
  const int maxDim = 3;
  const int maxlhsSize = maxNodesPerElement*maxNodesPerElement;
  const int maxrhsSize = maxNodesPerElement;

  /*// supplemental algorithm setup
  const size_t supplementalAlgSize = supplementalAlg_.size();
  for ( size_t i = 0; i < supplementalAlgSize; ++i )
    supplementalAlg_[i]->setup();*/

  // deal with state
/*  ScalarElemDiffusionFunctor * diffusionOperator;

  ScalarFieldType &scalarQNp1   = scalarQ_->field_of_state(stk::mesh::StateNP1);

  if (useCollocation_){
    diffusionOperator = new CollocationScalarElemDiffusionFunctor(bulk_data, meta_data,
      scalarQNp1, *diffFluxCoeff_, *coordinates_, nDim);
  }
  else*/
  //{
   //}

  // define some common selectors
  stk::mesh::Selector s_locally_owned_union = meta_data.locally_owned_part()
    & stk::mesh::selectUnion(partVec_) 
    & !(realm_.get_inactive_selector());

  stk::mesh::BucketVector const& elem_buckets =
    realm_.get_buckets( stk::topology::ELEMENT_RANK, s_locally_owned_union );

  const int bytes_per_team = maxNumScsIp * maxNodesPerElement * sizeof(double);
  // TODO: This may substantially overestimate the scratch space needed depending on what
  // element types are actually present. We should investigate whether the cost of this matters
  // and if so consider the Aria approach where a separate algorithm is created per topology.
  const int bytes_per_thread = (maxNodesPerElement + maxNodesPerElement + maxNodesPerElement*maxDim
      + maxNumScsIp*maxDim + maxDim*maxNumScsIp*maxNodesPerElement * maxDim*maxNumScsIp*maxNodesPerElement
      + maxNumScsIp + maxlhsSize + maxrhsSize)*sizeof(double)
      + maxNodesPerElement * sizeof(stk::mesh::Entity)
      + maxrhsSize*sizeof(int); // For TpetraLinearSystem::sumInto vector of localIds

  auto team_exec = get_team_policy(elem_buckets.size(), bytes_per_team, bytes_per_thread);

  Kokkos::parallel_for("Nalu::AssembleScalarElemDiffSolverAlgorithm::execute",
      team_exec, [&] (const DeviceTeam & team) {
      const int ib = team.league_rank();
      stk::mesh::Bucket & b = *elem_buckets[ib];
      const stk::mesh::Bucket::size_type length = b.size();

      // extract master element
      MasterElement *meSCS = realm_.get_surface_master_element(b.topology());
      const int nDim_ = nDim;
      const int nodesPerElement_ = meSCS->nodesPerElement_;
      const int numScsIp = meSCS->numIntPoints_;
      const int rhsSize = nodesPerElement_;
      const int lhsSize = nodesPerElement_*nodesPerElement_;

      SharedMemView<double**> shape_function_(team.team_shmem(), numScsIp, nodesPerElement_);

      // These are the per-thread handles. Better interface being worked on by Kokkos.
      SharedMemView<stk::mesh::Entity*> connected_nodes_;
      SharedMemView<double*> lhs_;
      SharedMemView<double*> rhs_;
      SharedMemView<double*> p_scalarQ;
      SharedMemView<double*> p_diffFluxCoeff;
      SharedMemView<double*> p_coordinates;
      SharedMemView<double*> p_scs_areav;
      SharedMemView<double*> p_dndx;
      SharedMemView<double*> p_deriv;
      SharedMemView<double*> p_det_j;
      SharedMemView<int*> localIdsScratch;
      {
        connected_nodes_ = Kokkos::subview(
            SharedMemView<stk::mesh::Entity**> (team.team_shmem(), team.team_size(), nodesPerElement_),
            team.team_rank(), Kokkos::ALL());
        lhs_ = Kokkos::subview(
            SharedMemView<double**>(team.team_shmem(), team.team_size(), lhsSize),
            team.team_rank(), Kokkos::ALL());
        rhs_ = Kokkos::subview(
            SharedMemView<double**>(team.team_shmem(), team.team_size(), rhsSize),
            team.team_rank(), Kokkos::ALL());
        p_scalarQ = Kokkos::subview(
            SharedMemView<double**> (team.team_shmem(), team.team_size(), nodesPerElement_),
            team.team_rank(), Kokkos::ALL());
        p_diffFluxCoeff = Kokkos::subview(
            SharedMemView<double**> (team.team_shmem(), team.team_size(), nodesPerElement_),
            team.team_rank(), Kokkos::ALL());
        p_coordinates = Kokkos::subview(
            SharedMemView<double**> (team.team_shmem(), team.team_size(), nodesPerElement_*nDim_),
            team.team_rank(), Kokkos::ALL());
        p_scs_areav = Kokkos::subview(
            SharedMemView<double**> (team.team_shmem(), team.team_size(), numScsIp*nDim_),
            team.team_rank(), Kokkos::ALL());
        p_dndx = Kokkos::subview(
            SharedMemView<double**> (team.team_shmem(), team.team_size(), nDim_*numScsIp*nodesPerElement_),
            team.team_rank(), Kokkos::ALL());
        p_deriv = Kokkos::subview(
            SharedMemView<double**> (team.team_shmem(), team.team_size(), nDim_*numScsIp*nodesPerElement_),
            team.team_rank(), Kokkos::ALL());
        p_det_j = Kokkos::subview(
            SharedMemView<double**> (team.team_shmem(), team.team_size(), numScsIp),
            team.team_rank(), Kokkos::ALL());
        localIdsScratch = Kokkos::subview(
            SharedMemView<int**> (team.team_shmem(), team.team_size(), nodesPerElement_),
            team.team_rank(), Kokkos::ALL());
      }

      Kokkos::single(Kokkos::PerTeam(team), [&]() {
        meSCS->shape_fcn(&shape_function_(0, 0));
      });
      team.team_barrier();

      auto lrscv = meSCS->adjacentNodes();

      Kokkos::parallel_for(Kokkos::TeamThreadRange(team, length), [&] (const size_t k) {
        const stk::mesh::Entity elem = b[k];

        // get nodes
        stk::mesh::Entity const * node_rels = bulk_data.begin_nodes(elem);

        // zero lhs/rhs
        for ( int p = 0; p < lhsSize; ++p )
          lhs_(p) = 0.0;
        for ( int p = 0; p < rhsSize; ++p )
          rhs_(p) = 0.0;

        for ( int ni = 0; ni < nodesPerElement_; ++ni ) {
          stk::mesh::Entity node = node_rels[ni];

          // set connected nodes
          connected_nodes_(ni) = node;

          const double * coords = stk::mesh::field_data(*coordinates_, node );

          // gather scalars
          p_scalarQ[ni] = *stk::mesh::field_data(*scalarQ_, node );
          p_diffFluxCoeff[ni] = *stk::mesh::field_data(*diffFluxCoeff_, node);

          // gather vectors
          const int offSet = ni*nDim_;
          for ( int j=0; j < nDim_; ++j ) {
            p_coordinates[offSet+j] = coords[j];
          }
        }

        // compute geometry
        double scs_error = 0.0;
        meSCS->determinant(1, &p_coordinates[0], &p_scs_areav[0], &scs_error);
        // compute dndx
        meSCS->grad_op(1, &p_coordinates[0], &p_dndx[0], &p_deriv[0], &p_det_j[0], &scs_error);

        // start assembly
        for ( int ip = 0; ip < numScsIp; ++ip ) {

          // left and right nodes for this ip
          const int il = lrscv[2*ip];
          const int ir = lrscv[2*ip+1];

          // corresponding matrix rows
          const int rowL = il*nodesPerElement_;
          const int rowR = ir*nodesPerElement_;

          // save off ip values; offset to Shape Function
          double muIp = 0.0;
          for ( int ic = 0; ic < nodesPerElement_; ++ic ) {
            const double r = shape_function_(ip, ic);
            muIp += r*p_diffFluxCoeff[ic];
          }

          double qDiff = 0.0;
          for ( int ic = 0; ic < nodesPerElement_; ++ic ) {

            // diffusion
            double lhsfacDiff = 0.0;
            const int offSetDnDx = nDim_*nodesPerElement_*ip + ic*nDim_;
            for ( int j = 0; j < nDim_; ++j ) {
              lhsfacDiff += -muIp*p_dndx[offSetDnDx+j]*p_scs_areav[ip*nDim_+j];
            }

            qDiff += lhsfacDiff*p_scalarQ[ic];

            // lhs; il then ir
            lhs_(rowL+ic) += lhsfacDiff;
            lhs_(rowR+ic) -= lhsfacDiff;
          }

          // rhs; il then ir
          rhs_(il) -= qDiff;
          rhs_(ir) += qDiff;

        }

        // call supplemental
        //for ( size_t i = 0; i < supplementalAlgSize; ++i )
        //  supplementalAlg_[i]->elem_execute( &lhs[0], &rhs[0], elem, meSCS, meSCV);

        apply_coeff(connected_nodes_, rhs_, lhs_, localIdsScratch, __FILE__);
      });
    });
}

} // namespace nalu
} // namespace Sierra
