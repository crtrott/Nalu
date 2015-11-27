/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#ifndef INCLUDE_KOKKOSINTERFACE_H_
#define INCLUDE_KOKKOSINTERFACE_H_

#include <Kokkos_Core.hpp>

namespace sierra {
namespace nalu {

using HostSpace = Kokkos::HostSpace;
using DeviceSpace = Kokkos::DefaultExecutionSpace;
using DeviceShmem = DeviceSpace::scratch_memory_space;
template <typename T>
using SharedMemView = Kokkos::View<T, Kokkos::LayoutRight, DeviceShmem, Kokkos::MemoryUnmanaged>;

using DeviceTeamPolicy = Kokkos::TeamPolicy<DeviceSpace>;
using DeviceTeam = DeviceTeamPolicy::member_type;

inline DeviceTeamPolicy get_team_policy(const size_t sz, const size_t bytes_per_team,
    const size_t bytes_per_thread)
{
  return DeviceTeamPolicy(sz, Kokkos::AUTO,
      Kokkos::Experimental::TeamScratchRequest<DeviceShmem>(bytes_per_team, bytes_per_thread));
}

}
}

#endif /* INCLUDE_KOKKOSINTERFACE_H_ */
