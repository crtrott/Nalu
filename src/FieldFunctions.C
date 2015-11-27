/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


#include <FieldFunctions.h>

#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Bucket.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/Selector.hpp>
#include <stk_mesh/base/GetBuckets.hpp>
#include <stk_mesh/base/Part.hpp>
#include <stk_mesh/base/Field.hpp>

#include <KokkosInterface.h>

#include <algorithm>

namespace sierra {
namespace nalu {
 
void field_axpby(
  const stk::mesh::MetaData & metaData,
  const stk::mesh::BulkData & bulkData,
  const double alpha,
  const stk::mesh::FieldBase & xField,
  const double beta,
  const stk::mesh::FieldBase & yField,
  const bool auraIsActive,
  const stk::topology::rank_t entityRankValue)
{
  // decide on selector
  const stk::mesh::Selector selector = auraIsActive 
    ? metaData.universal_part() &
    stk::mesh::selectField(xField) &
    stk::mesh::selectField(yField)
    : (metaData.locally_owned_part() | metaData.globally_shared_part()) &
    stk::mesh::selectField(xField) &
    stk::mesh::selectField(yField);
 
  stk::mesh::BucketVector const& buckets = bulkData.get_buckets( entityRankValue, selector );

  auto team_policy = get_team_policy(buckets.size(), 0, 0);
  Kokkos::parallel_for("Nalu::field_axpby", team_policy,
      [&] (const DeviceTeam & team) {
    stk::mesh::Bucket & b = *buckets[team.league_rank()];
    const stk::mesh::Bucket::size_type length = b.size();
    const size_t fieldSize = field_bytes_per_entity(xField, b) / sizeof(double);
    const unsigned kmax = length * fieldSize;
    const double * x = (double*)stk::mesh::field_data(xField, b);
    double * y = (double*)stk::mesh::field_data(yField, b);
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, kmax), [&](const size_t k) {
      y[k] = alpha * x[k] + beta*y[k];
    });
  });
}

void field_fill(
  const stk::mesh::MetaData & metaData,
  const stk::mesh::BulkData & bulkData,
  const double alpha,
  const stk::mesh::FieldBase & xField,
  const bool auraIsActive,
  const stk::topology::rank_t entityRankValue)
{
  // decide on selector
  const stk::mesh::Selector selector = auraIsActive 
    ? metaData.universal_part() &
    stk::mesh::selectField(xField)
    : (metaData.locally_owned_part() | metaData.globally_shared_part()) &
    stk::mesh::selectField(xField);

  stk::mesh::BucketVector const& buckets = bulkData.get_buckets( entityRankValue, selector );

  auto team_policy = get_team_policy(buckets.size(), 0, 0);
  Kokkos::parallel_for("Nalu::field_fill", team_policy,
      [&] (const DeviceTeam & team) {
    stk::mesh::Bucket & b = *buckets[team.league_rank()];
    const stk::mesh::Bucket::size_type length = b.size();
    const unsigned fieldSize = field_bytes_per_entity(xField, b) / sizeof(double);
    const unsigned kmax = length * fieldSize;
    double * x = (double*)stk::mesh::field_data(xField, b);
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, kmax), [&](const size_t k) {
      x[k] = alpha;
    });
  });
}

void field_scale(
  const stk::mesh::MetaData & metaData,
  const stk::mesh::BulkData & bulkData,
  const double alpha,
  const stk::mesh::FieldBase & xField,
  const bool auraIsActive,
  const stk::topology::rank_t entityRankValue)
{
  // decide on selector
  const stk::mesh::Selector selector = auraIsActive 
    ? metaData.universal_part() &
    stk::mesh::selectField(xField)
    : (metaData.locally_owned_part() | metaData.globally_shared_part()) &
    stk::mesh::selectField(xField);

  stk::mesh::BucketVector const& buckets = bulkData.get_buckets( entityRankValue, selector );

  auto team_policy = get_team_policy(buckets.size(), 0, 0);
  Kokkos::parallel_for("Nalu::field_scale", team_policy,
      [&] (const DeviceTeam & team) {
    stk::mesh::Bucket & b = *buckets[team.league_rank()];
    const stk::mesh::Bucket::size_type length = b.size();
    const unsigned fieldSize = field_bytes_per_entity(xField, b) / sizeof(double);
    const unsigned kmax = length * fieldSize;
    double * x = (double*)stk::mesh::field_data(xField, b);
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, kmax), [&](const size_t k) {
      x[k] = alpha * x[k];
    });
  });
}

void field_copy(
  const stk::mesh::MetaData & metaData,
  const stk::mesh::BulkData & bulkData,
  const stk::mesh::FieldBase & xField,
  const stk::mesh::FieldBase & yField,
  const bool auraIsActive,
  const stk::topology::rank_t entityRankValue)
{
  // decide on selector
  const stk::mesh::Selector selector = auraIsActive 
    ? metaData.universal_part() &
    stk::mesh::selectField(xField) &
    stk::mesh::selectField(yField)
    : (metaData.locally_owned_part() | metaData.globally_shared_part()) &
    stk::mesh::selectField(xField) &
    stk::mesh::selectField(yField);

  stk::mesh::BucketVector const& buckets = bulkData.get_buckets( entityRankValue, selector );

  auto team_policy = get_team_policy(buckets.size(), 0, 0);
  Kokkos::parallel_for("Nalu::field_copy", team_policy,
      [&] (const DeviceTeam & team) {
    stk::mesh::Bucket & b = *buckets[team.league_rank()];
    const stk::mesh::Bucket::size_type length = b.size();
    const size_t fieldSize = field_bytes_per_entity(xField, b) / sizeof(double);
    const unsigned kmax = length * fieldSize;
    const double * x = (double*)stk::mesh::field_data(xField, b);
    double * y = (double*)stk::mesh::field_data(yField, b);
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, kmax), [&](const size_t k) {
      y[k] = x[k];
    });
  });

}

void field_index_copy(
  const stk::mesh::MetaData & metaData,
  const stk::mesh::BulkData & bulkData,
  const stk::mesh::FieldBase & xField,
  const int xFieldIndex,
  const stk::mesh::FieldBase & yField,
  const int yFieldIndex,
  const bool auraIsActive,
  const stk::topology::rank_t entityRankValue)
{
  // decide on selector
  const stk::mesh::Selector selector = auraIsActive 
    ? metaData.universal_part() &
    stk::mesh::selectField(xField) &
    stk::mesh::selectField(yField)
    : (metaData.locally_owned_part() | metaData.globally_shared_part()) &
    stk::mesh::selectField(xField) &
    stk::mesh::selectField(yField);

  stk::mesh::BucketVector const& buckets = bulkData.get_buckets( entityRankValue, selector );

  auto team_policy = get_team_policy(buckets.size(), 0, 0);
  Kokkos::parallel_for("Nalu::field_index_copy", team_policy,
      [&] (const DeviceTeam & team) {
    stk::mesh::Bucket & b = *buckets[team.league_rank()];
    const stk::mesh::Bucket::size_type length = b.size();
    const size_t xFieldSize = field_bytes_per_entity(xField, b) / sizeof(double);
    const size_t yFieldSize = field_bytes_per_entity(yField, b) / sizeof(double);
    const double * x = (double*)stk::mesh::field_data(xField, b);
    double * y = (double*)stk::mesh::field_data(yField, b);
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, length), [&](const size_t k) {
      y[k*yFieldSize+yFieldIndex] = x[k*xFieldSize+xFieldIndex];
    });
  });

}

} // namespace nalu
} // namespace Sierra
