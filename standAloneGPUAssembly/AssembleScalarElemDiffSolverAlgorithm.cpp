/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


// nalu
#include <iostream>
#include <fstream>
#include <KokkosInterface.h>
#include <HexSCS.h>

namespace sierra{
namespace nalu{

template <typename T>
using AllTeamThreadsView = Kokkos::View<T, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

void execute()
{
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
  Kokkos::View<unsigned **,Kokkos::LayoutRight> inConnectivity("inConnectivity", inNumElems, 8);
  Kokkos::View<unsigned *> inBucketNumElems("inBucketNumElems", inNumBuckets);
  Kokkos::View<unsigned **,Kokkos::LayoutRight> inBucketElemLocalIds("inBucketElemLocalIds", inNumBuckets, 512);
  Kokkos::View<double **,Kokkos::LayoutRight> inCoords("inCoords", inTotalNodes, nDim);
  Kokkos::View<double *> inScalarQ("inScalarQ", inTotalNodes);
  Kokkos::View<double *> inDiffFluxCoeff("inDiffFluxCoeff", inTotalNodes);
  Kokkos::View<double **,Kokkos::LayoutRight> inLhsOut("inLhsOut", inNumElems, maxNodesPerElement*maxNodesPerElement);
  Kokkos::View<double **,Kokkos::LayoutRight> inRhsOut("inRhsOut", inNumElems, maxNodesPerElement);


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

  //auto team_exec = get_team_policy(inNumBuckets, bytes_per_team, bytes_per_thread);
  auto team_exec = get_team_policy(inNumBuckets, 1024, 0);

  Kokkos::View<double**,Kokkos::LayoutRight> scratch_view("ScratchView",64*16,(bytes_per_team+512*bytes_per_thread)/8);
  Kokkos::View<int*> lock_array("ScratchLockArray",64*16);

  std::cout<< "SharedMemory: " << bytes_per_team << " " << bytes_per_thread << " " << scratch_view.dimension_1()<<std::endl;
  Kokkos::parallel_for("Nalu::AssembleScalarElemDiffSolverAlgorithm::execute",
      team_exec, KOKKOS_LAMBDA (const DeviceTeam & team) {

      const int ib = team.league_rank();
      const unsigned length = inBucketNumElems(ib);

      int my_scratch_index = -1;
 
      
      /*Kokkos::single(Kokkos::PerTeam(team), [&] (int& idx) {
        idx = team.league_rank()%(16*16);
        while(!Kokkos::atomic_compare_exchange_strong(&lock_array(idx),0,1)) {
          idx++;
          if(idx==16*16) idx = 0;
        }
      },my_scratch_index);*/
      
      int idx = -1;
      SharedMemView<int> index(team.team_scratch(1));
      if(team.team_rank()==0) {
                idx = team.league_rank()%(64*16);
        while(!Kokkos::atomic_compare_exchange_strong(&lock_array(idx),0,1)) {
          idx++;
          if(idx==64*16) idx = 0;
        }
        index() = idx;
      }
      team.team_barrier();
      my_scratch_index = index();
      //team.team_broadcast(my_scratch_index,0);
      //printf("Team: %i Rank %i HdwId: %i ScratchIndex: %i\n",team.league_rank(),team.team_rank(),Kokkos::OpenMP::hardware_thread_id(),my_scratch_index);
      //team.team_barrier();
 
      // extract master element
      constexpr int nDim_ = 3;
      constexpr int nodesPerElement_ = 8;
      constexpr int numScsIp = 12;
      const int rhsSize = nodesPerElement_;
      const int lhsSize = nodesPerElement_*nodesPerElement_;


      /*Kokkos::View<double*,Kokkos::LayoutRight,Kokkos::MemoryTraits<Kokkos::Unmanaged>>
        my_scratch(scratch_view,my_scratch_index,Kokkos::ALL());
      int offset = 0;

      SharedMemView<double**> shape_function_((double*)&scratch_view(my_scratch_index,offset),numScsIp,nodesPerElement_);
      offset += numScsIp*nodesPerElement_;

      // These are the per-thread handles. Better interface being worked on by Kokkos.
      int off_increment = lhsSize;
      SharedMemView<double*> lhs_((double*)&my_scratch(offset + team.team_rank()*off_increment),lhsSize);
      offset += off_increment*team.team_size();
      //printf("Team: %i Rank %i ScratchIndex: %i Pointer %p %p\n",team.league_rank(),team.team_rank(),my_scratch_index,shape_function_.data(),lhs_.data());

      off_increment = rhsSize;
      SharedMemView<double*> rhs_((double*)&my_scratch(offset + team.team_rank()*off_increment),rhsSize);
      offset += off_increment*team.team_size();

      off_increment = nodesPerElement_;
      SharedMemView<double*> p_scalarQ((double*)&my_scratch(offset + team.team_rank()*off_increment),nodesPerElement_);
      offset += off_increment*team.team_size();

      off_increment = nodesPerElement_;
      SharedMemView<double*> p_diffFluxCoeff((double*)&my_scratch(offset + team.team_rank()*off_increment),nodesPerElement_);
      offset += off_increment*team.team_size();

      off_increment = nodesPerElement_*3;
      SharedMemView<double*[3]> p_coordinates((double*)&my_scratch(offset + team.team_rank()*off_increment),nodesPerElement_);
      offset += off_increment*team.team_size();

      off_increment = numScsIp*3;
      SharedMemView<double*[3]> p_scs_areav((double*)&my_scratch(offset + team.team_rank()*off_increment),numScsIp);
      offset += off_increment*team.team_size();

      off_increment = numScsIp*8*3;
      SharedMemView<double*[8][3]> p_dndx((double*)&my_scratch(offset + team.team_rank()*off_increment),numScsIp);
      offset += off_increment*team.team_size();

      off_increment = numScsIp*8*3;
      SharedMemView<double*[8][3]> p_deriv((double*)&my_scratch(offset + team.team_rank()*off_increment),numScsIp);
      offset += off_increment*team.team_size();

      off_increment = numScsIp;
      SharedMemView<double*> p_det_j((double*)&my_scratch(offset + team.team_rank()*off_increment),numScsIp);
      offset += off_increment*team.team_size();

      off_increment = rhsSize/2;
      SharedMemView<int*> localIdsScratch((int*)&my_scratch(offset + team.team_rank()*off_increment),rhsSize);
      offset += off_increment*team.team_size();*/


      int offset = 0;
      const int team_size = team.team_size();
      SharedMemView<double **> shape_function_((double*)&scratch_view(my_scratch_index, offset), numScsIp, nodesPerElement_);
      offset += numScsIp*nodesPerElement_;

      // These are the per-thread handles. Better interface being worked on by Kokkos.
      int off_increment = lhsSize;
      AllTeamThreadsView<double**> lhs_all((double*)&scratch_view(my_scratch_index, offset), team_size, lhsSize);
      auto lhs_ = Kokkos::subview(lhs_all, team.team_rank(), Kokkos::ALL);
      offset += off_increment*team.team_size();
      //printf("Team: %i Rank %i ScratchIndex: %i Pointer %p %p\n",team.league_rank(),team.team_rank(),my_scratch_index,shape_function_.data(),lhs_.data());

      off_increment = rhsSize;
      AllTeamThreadsView<double**> rhs_all((double*)&scratch_view(my_scratch_index, offset), team_size, rhsSize);
      auto rhs_ = Kokkos::subview(rhs_all, team.team_rank(), Kokkos::ALL);
      offset += off_increment*team.team_size();

      off_increment = nodesPerElement_;
      AllTeamThreadsView<double**> p_scalarQall((double*)&scratch_view(my_scratch_index, offset), team_size, nodesPerElement_);
      auto p_scalarQ = Kokkos::subview(p_scalarQall, team.team_rank(), Kokkos::ALL);
      offset += off_increment*team.team_size();

      off_increment = nodesPerElement_;
      AllTeamThreadsView<double**> p_diffFluxCoeffall((double*)&scratch_view(my_scratch_index, offset), team_size, nodesPerElement_);
      auto p_diffFluxCoeff = Kokkos::subview(p_diffFluxCoeffall, team.team_rank(), Kokkos::ALL);
      offset += off_increment*team.team_size();

      off_increment = nodesPerElement_*3;
      AllTeamThreadsView<double**[3]> p_coordinatesall((double*)&scratch_view(my_scratch_index, offset), team_size, nodesPerElement_);
      auto p_coordinates = Kokkos::subview(p_coordinatesall, team.team_rank(), Kokkos::ALL, Kokkos::ALL);
      offset += off_increment*team.team_size();

      off_increment = numScsIp*3;
      AllTeamThreadsView<double**[3]> p_scs_areavall((double*)&scratch_view(my_scratch_index, offset), team_size, numScsIp);
      auto p_scs_areav = Kokkos::subview(p_scs_areavall, team.team_rank(), Kokkos::ALL, Kokkos::ALL);
      offset += off_increment*team.team_size();

      off_increment = numScsIp*8*3;
      AllTeamThreadsView<double**[8][3]> p_dndxall((double*)&scratch_view(my_scratch_index, offset), team_size, numScsIp);
      auto p_dndx = Kokkos::subview(p_dndxall, team.team_rank(), Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
      offset += off_increment*team.team_size();

      off_increment = numScsIp*8*3;
      AllTeamThreadsView<double**[8][3]> p_derivall((double*)&scratch_view(my_scratch_index, offset), team_size, numScsIp);
      auto p_deriv = Kokkos::subview(p_derivall, team.team_rank(), Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
      offset += off_increment*team.team_size();

      off_increment = numScsIp;
      AllTeamThreadsView<double**> p_det_jall((double*)&scratch_view(my_scratch_index, offset), team_size, numScsIp);
      auto p_det_j = Kokkos::subview(p_det_jall, team.team_rank(), Kokkos::ALL);
      offset += off_increment*team.team_size();

      off_increment = rhsSize/2;
      AllTeamThreadsView<int**> localIdsScratchall((int*)&scratch_view(my_scratch_index, offset), team_size, rhsSize);
      auto localIdsScratch = Kokkos::subview(localIdsScratchall, team.team_rank(), Kokkos::ALL);
      offset += off_increment*team.team_size();

      Kokkos::single(Kokkos::PerTeam(team), [&]() {
        //meSCS->shape_fcn(&shape_function_(0, 0));
        hex_shape_fcn(shape_function_);
      });
      team.team_barrier();
constexpr int lrscv[24] = {
  0, 1,
  1, 2,
  2, 3,
  0, 3,
  4, 5,
  5, 6,
  6, 7,
  4, 7,
  0, 4,
  1, 5,
  2, 6,
  3, 7
};

      //auto lrscv = hex_scs_adjacent_nodes;

      Kokkos::parallel_for(Kokkos::TeamThreadRange(team, length), [&] (const int k) {

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

      team.team_barrier();
      Kokkos::single(Kokkos::PerTeam(team), [&] () {
        Kokkos::atomic_exchange(&lock_array(my_scratch_index),0);
      });
      team.team_barrier();
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
        std::cout << "Error in rhs " << i << ", " << j << " Result: " << rhsOut(i,j) << " " << inRhsOut(i,j) << std::endl;
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
