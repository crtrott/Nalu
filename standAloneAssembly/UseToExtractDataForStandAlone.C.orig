/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


// nalu
#include <fstream>
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

  Kokkos::View<unsigned*, Kokkos::LayoutRight> bucketNumElems("bucketNumElems", elem_buckets.size());
  Kokkos::View<unsigned**, Kokkos::LayoutRight> bucketElemLocalIds("bucketElemLocalIds", elem_buckets.size(), 512);
  unsigned numElems =0;
  for (int ib = 0; ib < elem_buckets.size(); ++ib)
  {
    auto b = elem_buckets[ib];
    numElems += b->size();
    bucketNumElems(ib) = b->size();
  }

  Kokkos::View<unsigned**, Kokkos::LayoutRight> elemConnectivity("elemConnectivity", numElems, 8);

  for (int ib = 0; ib < elem_buckets.size(); ++ib)
  {
    auto & bucket = *elem_buckets[ib];
    const stk::mesh::Bucket::size_type length = bucket.size();
    for (unsigned i = 0; i < length; ++i)
    {
      auto elem = bucket[i];
      auto elemLocalId = bulk_data.local_id(elem);
      bucketElemLocalIds(ib, i) = elemLocalId;
      stk::mesh::Entity const * node_rels = bulk_data.begin_nodes(elem);
      for(int ni = 0; ni < 8; ++ni)
      {
        stk::mesh::Entity node = node_rels[ni];
        const double * coords = stk::mesh::field_data(*coordinates_, node);

        elemConnectivity(elemLocalId, ni) = bulk_data.local_id(node);

      }
    }
  }
  unsigned totalNodes = 0;
  auto & nodeBuckets = realm_.get_buckets( stk::topology::NODE_RANK, s_locally_owned_union);
  for (auto && nodeBucket : nodeBuckets)
  {
    totalNodes += nodeBucket->size();
  }

  Kokkos::View<double*, Kokkos::LayoutRight> diffFluxCoeff("diffFluxCoeff", totalNodes);
  Kokkos::View<double*, Kokkos::LayoutRight> scalarQ("scalarQ", totalNodes);
  Kokkos::View<double**, Kokkos::LayoutRight> coords("coords", totalNodes, nDim);
  for (auto && nodeBucket : nodeBuckets)
  {
    for (auto && node : *nodeBucket)
    {
      auto nodeId = bulk_data.local_id(node);
      ThrowRequire(nodeId >= 0 && nodeId < totalNodes);
      diffFluxCoeff(nodeId) = *stk::mesh::field_data(*diffFluxCoeff_, node);
      scalarQ(nodeId) = *stk::mesh::field_data(*scalarQ_, node);
      double * c = stk::mesh::field_data(*coordinates_, node );
      coords(nodeId, 0) = c[0];
      coords(nodeId, 1) = c[1];
      coords(nodeId, 2) = c[2];
    }
  }

  Kokkos::View<double**, Kokkos::LayoutRight> lhsOut("lhsOut", numElems, maxlhsSize);
  Kokkos::View<double**, Kokkos::LayoutRight> rhsOut("rhsOut", numElems, maxrhsSize);


  const int bytes_per_team = SharedMemView<double *>::shmem_size(maxNumScsIp * maxNodesPerElement);
  // TODO: This may substantially overestimate the scratch space needed depending on what
  // element types are actually present. We should investigate whether the cost of this matters
  // and if so consider the Aria approach where a separate algorithm is created per topology.
  const int bytes_per_thread =
      SharedMemView<double *>::shmem_size(maxNodesPerElement) +
      SharedMemView<double *>::shmem_size(maxNodesPerElement) +
      SharedMemView<double *>::shmem_size(maxNodesPerElement*maxDim) +
      SharedMemView<double *>::shmem_size(maxNumScsIp*maxDim) +
      SharedMemView<double *>::shmem_size(maxDim*maxNumScsIp*maxNodesPerElement) +
      SharedMemView<double *>::shmem_size(maxDim*maxNumScsIp*maxNodesPerElement) +
      SharedMemView<double *>::shmem_size(maxNumScsIp) +
      SharedMemView<double *>::shmem_size(maxlhsSize) +
      SharedMemView<double *>::shmem_size(maxrhsSize) +
      SharedMemView<stk::mesh::Entity *>::shmem_size(maxNodesPerElement) +
      SharedMemView<int *>::shmem_size(maxrhsSize); // For TpetraLinearSystem::sumInto vector of localIds

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

      const int scratch_level = 2;
      SharedMemView<double**> shape_function_(team.team_scratch(scratch_level), numScsIp, nodesPerElement_);

      // These are the per-thread handles. Better interface being worked on by Kokkos.
      SharedMemView<stk::mesh::Entity*> connected_nodes_(team.thread_scratch(scratch_level), nodesPerElement_);
      SharedMemView<double*> lhs_(team.thread_scratch(scratch_level), lhsSize);
      SharedMemView<double*> rhs_(team.thread_scratch(scratch_level), rhsSize);
      SharedMemView<double*> p_scalarQ(team.thread_scratch(scratch_level), nodesPerElement_);
      SharedMemView<double*> p_diffFluxCoeff(team.thread_scratch(scratch_level), nodesPerElement_);
      SharedMemView<double*> p_coordinates(team.thread_scratch(scratch_level), nodesPerElement_*nDim_);
      SharedMemView<double*> p_scs_areav(team.thread_scratch(scratch_level), numScsIp*nDim_);
      SharedMemView<double*> p_dndx(team.thread_scratch(scratch_level), nDim_*numScsIp*nodesPerElement_);
      SharedMemView<double*> p_deriv(team.thread_scratch(scratch_level), nDim_*numScsIp*nodesPerElement_);
      SharedMemView<double*> p_det_j(team.thread_scratch(scratch_level), numScsIp);
      SharedMemView<int*> localIdsScratch(team.thread_scratch(scratch_level), rhsSize);

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
        auto elemLocalId = bulk_data.local_id(elem);
        ThrowRequire(elemLocalId >= 0 && elemLocalId < numElems);
        for (int i = 0; i < rhsSize; ++i)
        {
          rhsOut(elemLocalId, i);
        }
        for (int i = 0; i < lhsSize; ++i)
        {
          lhsOut(elemLocalId, i);
        }
      });
    });

  std::ofstream file;
  auto numElemBuckets = elem_buckets.size();
  file.open("nalu_restart.bin", std::ios::binary);
  file.write( (char *) & numElems, sizeof(unsigned));
  file.write( (char *) & totalNodes, sizeof(unsigned));
  file.write( (char *) & numElemBuckets, sizeof(unsigned));
  file.write( (char *) bucketNumElems.ptr_on_device(), bucketNumElems.capacity() * sizeof(unsigned));
  file.write( (char *) bucketElemLocalIds.ptr_on_device(), bucketElemLocalIds.capacity() * sizeof(unsigned));
  file.write( (char *) elemConnectivity.ptr_on_device(), elemConnectivity.capacity() * sizeof(unsigned));
  file.write( (char *) coords.ptr_on_device(), coords.capacity() * sizeof(double));
  file.write( (char *) scalarQ.ptr_on_device(), scalarQ.capacity() * sizeof(double));
  file.write( (char *) diffFluxCoeff.ptr_on_device(), diffFluxCoeff.capacity() * sizeof(double));
  file.write( (char *) lhsOut.ptr_on_device(), lhsOut.capacity() * sizeof(double));
  file.write( (char *) rhsOut.ptr_on_device(), rhsOut.capacity() * sizeof(double));
  file.close();

  unsigned inNumElems;
  unsigned inTotalNodes;
  unsigned inNumBuckets;
  std::ifstream inFile;
  inFile.open("nalu_restart.bin", std::ios::binary);
  inFile.read( (char *) & inNumElems, sizeof(unsigned));
  inFile.read( (char *) & inTotalNodes, sizeof(unsigned));
  inFile.read( (char *) & inNumBuckets, sizeof(unsigned));
  //
  Kokkos::View<unsigned **> inConnectivity("inConnectivity", inNumElems, 8);
  Kokkos::View<unsigned *> inBucketNumElems("inBucketNumElems", inNumBuckets);
  Kokkos::View<unsigned **> inBucketElemLocalIds("inBucketElemLocalIds", inNumBuckets, 512);
  Kokkos::View<double **> inCoords("inCoords", inTotalNodes, nDim);
  Kokkos::View<double *> inScalarQ("inScalarQ", inTotalNodes);
  Kokkos::View<double *> inDiffFluxCoeff("inDiffFluxCoeff", inTotalNodes);
  Kokkos::View<double **> inLhsOut("inLhsOut", numElems, maxNodesPerElement*maxNodesPerElement);
  Kokkos::View<double **> inRhsOut("inRhsOut", numElems, maxNodesPerElement);


  inFile.read( (char *) inBucketNumElems.ptr_on_device(), inBucketNumElems.capacity() * sizeof(unsigned));
  inFile.read( (char *) inBucketElemLocalIds.ptr_on_device(), inBucketElemLocalIds.capacity() * sizeof(unsigned));
  inFile.read( (char *) inConnectivity.ptr_on_device(), inConnectivity.capacity() * sizeof(unsigned));
  inFile.read( (char *) inCoords.ptr_on_device(), inCoords.capacity() * sizeof(double));
  inFile.read( (char *) inScalarQ.ptr_on_device(), inScalarQ.capacity() * sizeof(double));
  inFile.read( (char *) inDiffFluxCoeff.ptr_on_device(), inDiffFluxCoeff.capacity() * sizeof(double));
  inFile.read( (char *) inLhsOut.ptr_on_device(), inLhsOut.capacity() * sizeof(double));
  inFile.read( (char *) inRhsOut.ptr_on_device(), inRhsOut.capacity() * sizeof(double));

  for (int i = 0; i < numElems; ++i)
  {
    for (int j = 0; j < 8; ++j)
    {
      ThrowRequire(elemConnectivity(i, j) == inConnectivity(i, j));
      ThrowRequire(rhsOut(i, j) == inRhsOut(i, j));
    }
    for (int j = 0; j < maxNodesPerElement*maxNodesPerElement; ++j)
    {
      ThrowRequire(lhsOut(i, j) == inLhsOut(i, j));
    }
  }
  for (int i = 0; i < totalNodes; ++i)
  {
    ThrowRequire(coords(i, 0) == inCoords(i, 0));
    ThrowRequire(coords(i, 1) == inCoords(i, 1));
    ThrowRequire(coords(i, 2) == inCoords(i, 2));
    ThrowRequire(scalarQ(i) == inScalarQ(i));
    ThrowRequire(diffFluxCoeff(i) == inDiffFluxCoeff(i));
  }
  for (int i = 0; i < inNumBuckets; ++i)
  {
    ThrowRequire(bucketNumElems(i) == inBucketNumElems(i));
    for (int j = 0; j < 512; ++j)
    {
      ThrowRequire(bucketElemLocalIds(i,j) == bucketElemLocalIds(i,j));
    }

  }
  inFile.close();
}

} // namespace nalu
} // namespace Sierra
