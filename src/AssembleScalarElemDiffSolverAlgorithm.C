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

#include <Kokkos_Core.hpp>

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

  const int numWorkBuckets = 32;
  const int maxElementsPerBucket = 512;
  const int maxNodesPerElement = 8;
  const int maxNumScsIp = 16;
  const int lhsSize = maxNodesPerElement*maxNodesPerElement;
  const int rhsSize = maxNodesPerElement;
//  Kokkos::View<double******> lhs("lhsScratch", numWorkBuckets, maxElementsPerBucket,
//      maxNodesPerElement, maxNodesPerElement, sizeOfSystem_, sizeOfSystem_);
  Kokkos::View<double***, Kokkos::LayoutRight> lhsScratch("lhsScratch", numWorkBuckets, maxElementsPerBucket, lhsSize);
  Kokkos::View<double***, Kokkos::LayoutRight> rhsScratch("rhsScratch", numWorkBuckets, maxElementsPerBucket, rhsSize);
  Kokkos::View<stk::mesh::Entity***, Kokkos::LayoutRight> connectedNodesScratch("cnScratch", numWorkBuckets, maxElementsPerBucket, maxNodesPerElement);
  Kokkos::View<double***, Kokkos::LayoutRight> shapeFunctionScratch("shapeFcnScratch", numWorkBuckets, maxNumScsIp, maxNodesPerElement);

  /*// supplemental algorithm setup
  const size_t supplementalAlgSize = supplementalAlg_.size();
  for ( size_t i = 0; i < supplementalAlgSize; ++i )
    supplementalAlg_[i]->setup();*/

  // deal with state
  ScalarFieldType &scalarQNp1   = scalarQ_->field_of_state(stk::mesh::StateNP1);

/*  ScalarElemDiffusionFunctor * diffusionOperator;

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

  typedef typename Kokkos::TeamPolicy<Kokkos::Serial>::member_type team_type;
  for (unsigned bucketOffset = 0; bucketOffset < elem_buckets.size(); bucketOffset += numWorkBuckets)
  {
    const int bucketEnd = std::min(bucketOffset + numWorkBuckets, (unsigned int)elem_buckets.size());
    Kokkos::parallel_for("Nalu::AssembleScalarElemDiffSolverAlgorithm::execute",
        Kokkos::TeamPolicy<Kokkos::Serial>(bucketEnd-bucketOffset, Kokkos::AUTO), [&] (const team_type& team) {
      const int ib = team.league_rank();
      stk::mesh::Bucket & b = *elem_buckets[bucketOffset+ib];
      const stk::mesh::Bucket::size_type length = b.size();

      // extract master element
      MasterElement *meSCS = realm_.get_surface_master_element(b.topology());

      CVFEMScalarElemDiffusionFunctor diffusionOperator(bulk_data, meta_data,
          scalarQNp1, *diffFluxCoeff_, *coordinates_, nDim, b, *meSCS,
          Kokkos::subview(lhsScratch, ib, Kokkos::ALL(), Kokkos::ALL()),
          Kokkos::subview(rhsScratch, ib, Kokkos::ALL(), Kokkos::ALL()),
          Kokkos::subview(connectedNodesScratch, ib, Kokkos::ALL(), Kokkos::ALL()),
          Kokkos::subview(shapeFunctionScratch, ib, Kokkos::ALL(), Kokkos::ALL())
          );

      Kokkos::parallel_for(Kokkos::TeamThreadRange(team, length), [&] (const size_t k) {
        diffusionOperator(k);

        // get elem
        stk::mesh::Entity elem = b[k];
        // call supplemental
        //for ( size_t i = 0; i < supplementalAlgSize; ++i )
        //  supplementalAlg_[i]->elem_execute( &lhs[0], &rhs[0], elem, meSCS, meSCV);

        apply_coeff(Kokkos::subview(connectedNodesScratch, ib, k, Kokkos::ALL()),
            Kokkos::subview(rhsScratch, ib, k, Kokkos::ALL()),
            Kokkos::subview(lhsScratch, ib, k, Kokkos::ALL()),
            __FILE__);
      });
    });
  }
}

} // namespace nalu
} // namespace Sierra
