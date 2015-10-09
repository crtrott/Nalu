/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


// nalu
#include <AssembleNodalGradElemAlgorithm.h>
#include <Algorithm.h>

#include <FieldTypeDef.h>
#include <Realm.h>
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

struct nodalGradientElem{
private:

  //Bucket and Element Data
  const stk::mesh::Bucket & b_;
  MasterElement & meSCS_;
  Kokkos::View<const double **, Kokkos::LayoutRight> shape_function_;

  //InputFields
  ScalarFieldType & scalarQ_;
  ScalarFieldType & dualNodalVolume_;
  VectorFieldType & coordinates_;

  //OutputFields
  VectorFieldType & dqdx_;

  //Parameters
  const int *lrscv;
  const int nDim_;
  const int numScsIp_;
  const int nodesPerElement_;

public:
  nodalGradientElem(const stk::mesh::Bucket & b, MasterElement & meSCS,
      Kokkos::View<const double **, Kokkos::LayoutRight> p_shape_function,
      ScalarFieldType & scalarQ, VectorFieldType & dqdx,
      ScalarFieldType & dualNodalVolume, VectorFieldType & coordinates,
      int nDim):
      b_(b),
      meSCS_(meSCS),
      shape_function_(p_shape_function),
      scalarQ_(scalarQ),
      dualNodalVolume_(dualNodalVolume),
      coordinates_(coordinates),
      dqdx_(dqdx),
      nDim_(nDim),
      numScsIp_(meSCS_.numIntPoints_),
      nodesPerElement_(meSCS_.nodesPerElement_)
  {
    lrscv = meSCS_.adjacentNodes();
  }
  void operator()(stk::mesh::Bucket::size_type elem_offset){
    // get elem
    //===============================================
    // gather nodal data; this is how we do it now..
    //===============================================
    stk::mesh::Entity const * node_rels = b_.begin_nodes(elem_offset);
    const int num_nodes = b_.num_nodes(elem_offset);

    // temporary arrays
    double p_scalarQ[nodesPerElement_];
    double p_dualVolume[nodesPerElement_];
    double p_coordinates[nodesPerElement_*nDim_];
    double p_scs_areav[numScsIp_*nDim_];

    for ( int ni = 0; ni < num_nodes; ++ni ) {
      stk::mesh::Entity node = node_rels[ni];

      const double * coords = stk::mesh::field_data(coordinates_, node );

      // gather scalars
      p_scalarQ[ni]    = *stk::mesh::field_data(scalarQ_, node);
      p_dualVolume[ni] = *stk::mesh::field_data(dualNodalVolume_, node);

      // gather vectors
      const int offSet = ni*nDim_;
      for ( int j=0; j < nDim_; ++j ) {
        p_coordinates[offSet+j] = coords[j];
      }
    }

    // compute geometry
    double scs_error = 0.0;
    meSCS_.determinant(1, &p_coordinates[0], &p_scs_areav[0], &scs_error);

    // start assembly
    for ( int ip = 0; ip < numScsIp_; ++ip ) {

      // left and right nodes for this ip
      const int il = lrscv[2*ip];
      const int ir = lrscv[2*ip+1];

      stk::mesh::Entity nodeL = node_rels[il];
      stk::mesh::Entity nodeR = node_rels[ir];

      // pointer to fields to assemble
      double *gradQL = stk::mesh::field_data(dqdx_, nodeL );
      double *gradQR = stk::mesh::field_data(dqdx_, nodeR );

      // interpolate to scs point; operate on saved off ws_field
      double qIp = 0.0;
      const int offSet = ip*nodesPerElement_;
      for ( int ic = 0; ic < nodesPerElement_; ++ic ) {
        qIp += shape_function_(ip, ic) * p_scalarQ[ic];
      }

      // left and right volume
      double inv_volL = 1.0/p_dualVolume[il];
      double inv_volR = 1.0/p_dualVolume[ir];

      // assemble to il/ir
      for ( int j = 0; j < nDim_; ++j ) {
        double fac = qIp*p_scs_areav[ip*nDim_+j];
        Kokkos::atomic_add(&gradQL[j], fac*inv_volL);
        Kokkos::atomic_sub(&gradQR[j], fac*inv_volR);
      }
    }
   }
};

//==========================================================================
// Class Definition
//==========================================================================
// AssembleNodalGradElemAlgorithm - Green-Gauss gradient
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
AssembleNodalGradElemAlgorithm::AssembleNodalGradElemAlgorithm(
  Realm &realm,
  stk::mesh::Part *part,
  ScalarFieldType *scalarQ,
  VectorFieldType *dqdx,
  const bool useShifted)
  : Algorithm(realm, part),
    scalarQ_(scalarQ),
    dqdx_(dqdx),
    dualNodalVolume_(NULL),
    coordinates_(NULL),
    useShifted_(useShifted)
{
  // extract fields
  stk::mesh::MetaData & meta_data = realm_.meta_data();

  dualNodalVolume_ = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "dual_nodal_volume");
  coordinates_ = meta_data.get_field<VectorFieldType>(stk::topology::NODE_RANK, realm_.get_coordinates_name());
}

//--------------------------------------------------------------------------
//-------- execute ---------------------------------------------------------
//--------------------------------------------------------------------------
void
AssembleNodalGradElemAlgorithm::execute()
{
  stk::mesh::MetaData & meta_data = realm_.meta_data();

  const int nDim = meta_data.spatial_dimension();
  const int numWorkBuckets = 32;
  const int maxNumScsIp = 12;
  const int maxNodesPerElement = 8;
  Kokkos::View<double ***, Kokkos::LayoutRight> ws_shape_function("wsShapeFcn",
      numWorkBuckets, maxNumScsIp, maxNodesPerElement);

  // define some common selectors
  stk::mesh::Selector s_locally_owned_union = meta_data.locally_owned_part()
    & stk::mesh::selectUnion(partVec_) 
    & !(realm_.get_inactive_selector());

  stk::mesh::BucketVector const& elem_buckets =
    realm_.get_buckets( stk::topology::ELEMENT_RANK, s_locally_owned_union );

  typedef typename Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>::member_type team_type;

  const int numBuckets = elem_buckets.size();
  for(int bucketOffset = 0; bucketOffset < numBuckets; bucketOffset += numWorkBuckets)
  {
    const int bucketsThisPass = std::min(numWorkBuckets, numBuckets - bucketOffset);
    Kokkos::parallel_for("Nalu::AssembleNodalGradElemAlgorithm::execute()",
      Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>(bucketsThisPass, Kokkos::AUTO()), [&](team_type team) {
        const int ib = team.league_rank();
        const stk::mesh::Bucket & b = *elem_buckets[ib + bucketOffset];
        const auto length = b.size();

        MasterElement *meSCS = realm_.get_surface_master_element(b.topology());
        const int nodesPerElement = meSCS->nodesPerElement_;
        const int numScsIp = meSCS->numIntPoints_;
        if ( useShifted_ )
          meSCS->shifted_shape_fcn(&ws_shape_function(ib, 0, 0));
        else
          meSCS->shape_fcn(&ws_shape_function(ib, 0, 0));

        auto bucket_shape_function = Kokkos::subview(ws_shape_function, ib, Kokkos::ALL(), Kokkos::ALL());
        nodalGradientElem nodeGradFunctor(b, *meSCS, bucket_shape_function,
            *scalarQ_, *dqdx_, *dualNodalVolume_, *coordinates_, nDim);

        Kokkos::parallel_for(Kokkos::TeamThreadRange(team, length), [&] (const size_t k) {
          nodeGradFunctor(k);
        });
    });
  }
}




} // namespace nalu
} // namespace Sierra
