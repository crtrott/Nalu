/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


// nalu
#include <iostream>
#include <fstream>
#include <MasterElement.h>
#include <KokkosInterface.h>
#include <HexSCS.h>

namespace sierra{
namespace nalu{

void execute()
{
  HexSCS hexScsME;
  MasterElement * meSCS = &hexScsME;
  const int nDim = 3;

  const int maxNodesPerElement = 8;
  const int maxNumScsIp = 16;
  const int maxDim = 3;
  const int maxlhsSize = maxNodesPerElement*maxNodesPerElement;
  const int maxrhsSize = maxNodesPerElement;

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
  Kokkos::View<double **> inLhsOut("inLhsOut", inNumElems, maxNodesPerElement*maxNodesPerElement);
  Kokkos::View<double **> inRhsOut("inRhsOut", inNumElems, maxNodesPerElement);


  inFile.read( (char *) inBucketNumElems.ptr_on_device(), inBucketNumElems.capacity() * sizeof(unsigned));
  inFile.read( (char *) inBucketElemLocalIds.ptr_on_device(), inBucketElemLocalIds.capacity() * sizeof(unsigned));
  inFile.read( (char *) inConnectivity.ptr_on_device(), inConnectivity.capacity() * sizeof(unsigned));
  inFile.read( (char *) inCoords.ptr_on_device(), inCoords.capacity() * sizeof(double));
  inFile.read( (char *) inScalarQ.ptr_on_device(), inScalarQ.capacity() * sizeof(double));
  inFile.read( (char *) inDiffFluxCoeff.ptr_on_device(), inDiffFluxCoeff.capacity() * sizeof(double));
  inFile.read( (char *) inLhsOut.ptr_on_device(), inLhsOut.capacity() * sizeof(double));
  inFile.read( (char *) inRhsOut.ptr_on_device(), inRhsOut.capacity() * sizeof(double));


  inFile.close();

  Kokkos::View<double**, Kokkos::LayoutRight> lhsOut("lhsOut", inNumElems, maxlhsSize);
  Kokkos::View<double**, Kokkos::LayoutRight> rhsOut("rhsOut", inNumElems, maxrhsSize);


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
      SharedMemView<int *>::shmem_size(maxrhsSize); // For TpetraLinearSystem::sumInto vector of localIds

  auto team_exec = get_team_policy(inNumBuckets, bytes_per_team, bytes_per_thread);

  Kokkos::parallel_for("Nalu::AssembleScalarElemDiffSolverAlgorithm::execute",
      team_exec, [&] (const DeviceTeam & team) {
      const int ib = team.league_rank();
      const unsigned length = inBucketNumElems(ib);

      // extract master element
      const int nDim_ = nDim;
      const int nodesPerElement_ = meSCS->nodesPerElement_;
      const int numScsIp = meSCS->numIntPoints_;
      const int rhsSize = nodesPerElement_;
      const int lhsSize = nodesPerElement_*nodesPerElement_;

      const int scratch_level = 2;
      SharedMemView<double**> shape_function_(team.team_scratch(scratch_level), numScsIp, nodesPerElement_);

      // These are the per-thread handles. Better interface being worked on by Kokkos.
      SharedMemView<double*> lhs_(team.thread_scratch(scratch_level), lhsSize);
      SharedMemView<double*> rhs_(team.thread_scratch(scratch_level), rhsSize);
      SharedMemView<double*> p_scalarQ(team.thread_scratch(scratch_level), nodesPerElement_);
      SharedMemView<double*> p_diffFluxCoeff(team.thread_scratch(scratch_level), nodesPerElement_);
      SharedMemView<double*[3]> p_coordinates(team.thread_scratch(scratch_level), nodesPerElement_);
      SharedMemView<double*[3]> p_scs_areav(team.thread_scratch(scratch_level), numScsIp);
      SharedMemView<double*[8][3]> p_dndx(team.thread_scratch(scratch_level), numScsIp);
      SharedMemView<double*[8][3]> p_deriv(team.thread_scratch(scratch_level), numScsIp);
      SharedMemView<double*> p_det_j(team.thread_scratch(scratch_level), numScsIp);
      SharedMemView<int*> localIdsScratch(team.thread_scratch(scratch_level), rhsSize);

      Kokkos::single(Kokkos::PerTeam(team), [&]() {
        meSCS->shape_fcn(&shape_function_(0, 0));
      });
      team.team_barrier();

      auto lrscv = meSCS->adjacentNodes();

      Kokkos::parallel_for(Kokkos::TeamThreadRange(team, length), [&] (const size_t k) {

        // zero lhs/rhs
        for ( int p = 0; p < lhsSize; ++p )
          lhs_(p) = 0.0;
        for ( int p = 0; p < rhsSize; ++p )
          rhs_(p) = 0.0;

        auto elemLocalId = inBucketElemLocalIds(ib, k);

        for ( int ni = 0; ni < nodesPerElement_; ++ni ) {

          auto nodeLocalId = inConnectivity(elemLocalId, ni);
          // set connected nodes
          const double * coords = & inCoords(nodeLocalId, 0);

          // gather scalars
          p_scalarQ[ni] = inScalarQ(nodeLocalId);
          p_diffFluxCoeff[ni] = inDiffFluxCoeff(nodeLocalId);

          // gather vectors
          const int offSet = ni*nDim_;
          for ( int j=0; j < nDim_; ++j ) {
            p_coordinates(ni, j) = coords[j];
          }
        }

        // compute geometry
        double scs_error = 0.0;
        hex_scs_det<8, 12>(p_coordinates, p_scs_areav);
        // compute dndx
        int numGradError;
        hex_derivative<8, 12>(p_deriv);
        hex_gradient_operator<8, 12>(p_deriv, p_coordinates, p_dndx, p_det_j, scs_error, numGradError);

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
            for ( int j = 0; j < nDim_; ++j ) {
              lhsfacDiff += -muIp*p_dndx(ip, ic, j)*p_scs_areav(ip, j);
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

//        apply_coeff(connected_nodes_, rhs_, lhs_, localIdsScratch, __FILE__);
        for (int i = 0; i < rhsSize; ++i)
        {
          rhsOut(elemLocalId, i) = rhs_(i);
        }
        for (int i = 0; i < lhsSize; ++i)
        {
          lhsOut(elemLocalId, i) = lhs_(i);
        }
      });
    });

  double rhsNorm = 0;
  double lhsNorm = 0;

  constexpr double tolerance = 1.e-16;
  for (int i = 0; i < inNumElems; ++i)
  {
    for (int j = 0; j < 8; ++j)
    {
      if(std::abs(rhsOut(i, j) - inRhsOut(i, j)) > tolerance)
      {
        std::cout << "Error in rhs " << i << ", " << j << std::endl;
      }
      rhsNorm += rhsOut(i,j)*rhsOut(i,j);
    }
    for (int j = 0; j < maxNodesPerElement*maxNodesPerElement; ++j)
    {
      if(std::abs(lhsOut(i, j) - inLhsOut(i, j)) > tolerance)
      {
        std::cout << "Error in lhs " << i << ", " << j << std::endl;
      }
      lhsNorm += lhsOut(i,j)*lhsOut(i,j);
    }
  }

  std::cout << "Rhs: " << rhsNorm << ", Lhs: " << lhsNorm << std::endl;
}

} // namespace nalu
} // namespace Sierra

int main(int argc, char * argv[])
{
  Kokkos::initialize(argc, argv);
  sierra::nalu::execute();
  Kokkos::finalize();

  return 0;
}
