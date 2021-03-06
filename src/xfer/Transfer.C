/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


#include <Realm.h>
#include <Realms.h>
#include <Simulation.h>
#include <NaluEnv.h>

// yaml for parsing..
#include <yaml-cpp/yaml.h>
#include <NaluParsing.h>
#include <NaluParsingHelper.h>
#include <master_element/MasterElement.h>

// stk_mesh/base/fem
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/GetBuckets.hpp>
#include <stk_util/parallel/Parallel.hpp>

#include <xfer/Transfer.h>
#include <xfer/Transfers.h>
#include <xfer/FromMesh.h>
#include <xfer/ToMesh.h>
#include <xfer/LinInterp.h>
#include <stk_transfer/GeometricTransfer.hpp>

// stk_search
#include <stk_search/SearchMethod.hpp>

// timing
#include <stk_util/environment/CPUTime.hpp>

namespace sierra{
namespace nalu{

//==========================================================================
// Class Definition
//==========================================================================
// Transfer - base class for Transfer
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
Transfer::Transfer(
  Transfers &transfers)
  : transfers_(transfers),
    couplingPhysicsSpecified_(false),
    transferVariablesSpecified_(false),
    couplingPhysicsName_("none"),
    fromRealm_(NULL),
    toRealm_(NULL),
    name_("none"),
    transferType_("none"),
    transferObjective_("multi_physics"),
    searchMethodName_("none")
{
  // nothing to do
}

//--------------------------------------------------------------------------
//-------- destructor ------------------------------------------------------
//--------------------------------------------------------------------------
Transfer::~Transfer()
{
  // nothing
}

//--------------------------------------------------------------------------
//-------- load ------------------------------------------------------------
//--------------------------------------------------------------------------
void
Transfer::load(const YAML::Node & node)
{

  node["name"] >> name_;
  node["type"] >> transferType_;
  if ( node.FindValue("objective") ) {
    node["objective"] >> transferObjective_;
  }

  if ( node.FindValue("coupling_physics") ) {
    node["coupling_physics"] >> couplingPhysicsName_;
    couplingPhysicsSpecified_ = true;
  }

  // realm names
  const YAML::Node & realmPair = node["realm_pair"];
  if ( realmPair.size() != 2 )
    throw std::runtime_error("need two realm pairs for xfer");
  realmPair[0] >> realmPairName_.first;
  realmPair[1] >> realmPairName_.second;

  // mesh part pairs
  const YAML::Node & meshPartPair = node["mesh_part_pair"];
  if ( meshPartPair.size() != 2 )
    throw std::runtime_error("need two mesh part pairs for xfer");
  meshPartPair[0] >> meshPartPairName_.first;
  meshPartPair[1] >> meshPartPairName_.second;

  // search method
  if ( node.FindValue("search_method") ) {
    node["search_method"] >> searchMethodName_;
  }

  // now possible field names
  const YAML::Node *y_vars = node.FindValue("transfer_variables");
  if (y_vars) {
    transferVariablesSpecified_ = true;
    std::string fromName, toName;
    size_t varSize = y_vars->size();
    for (size_t ioption = 0; ioption < varSize; ioption++) {
      const YAML::Node & y_var = (*y_vars)[ioption];
      size_t varPairSize = y_var.size();
      if ( varPairSize != 2 )
        throw std::runtime_error("need two field name pairs for xfer");
      y_var[0] >> fromName;
      y_var[1] >> toName;
      transferVariablesPairName_.push_back(std::make_pair(fromName, toName));
    }
    
    // warn the user...
    NaluEnv::self().naluOutputP0()
      << "Specifying the transfer variables requires expert understanding; consider using coupling_physics" << std::endl;
  }

  // sanity check
  if ( couplingPhysicsSpecified_ && transferVariablesSpecified_ )
    throw std::runtime_error("physics set and transfer variables specified; will go with variables specified");

  if ( !couplingPhysicsSpecified_ && !transferVariablesSpecified_ )
    throw std::runtime_error("neither physics set nor transfer variables specified");

  // now proceed with possible pre-defined transfers
  if ( !transferVariablesSpecified_ ) {
    // hard code for a single type of physics transfer
    if ( couplingPhysicsName_ == "fluids_cht" ) {
      // h
      std::pair<std::string, std::string> thePairH;
      std::string sameNameH = "heat_transfer_coefficient";
      thePairH = std::make_pair(sameNameH, sameNameH);
      transferVariablesPairName_.push_back(thePairH);
      // Too
      std::pair<std::string, std::string> thePairT;
      std::string sameNameT = "reference_temperature";
      thePairT = std::make_pair(sameNameT, sameNameT);
      transferVariablesPairName_.push_back(thePairT);
    }
    else if ( couplingPhysicsName_ == "fluids_robin" ) {
      // q
      std::pair<std::string, std::string> thePairQ;
      std::string sameNameQ = "normal_heat_flux";
      thePairQ = std::make_pair(sameNameQ, sameNameQ);
      transferVariablesPairName_.push_back(thePairQ);
      // Too
      std::pair<std::string, std::string> thePairT;
      std::string fluidsT  = "temperature";
      std::string thermalT = "reference_temperature";
      thePairT = std::make_pair(fluidsT, thermalT);
      transferVariablesPairName_.push_back(thePairT);
      // alpha
      std::pair<std::string, std::string> thePairA;
      std::string sameNameA = "robin_coupling_parameter";
      thePairA = std::make_pair(sameNameA, sameNameA);
      transferVariablesPairName_.push_back(thePairA);
    }
    else if ( couplingPhysicsName_ == "thermal_cht" ) {
      // T -> T
      std::pair<std::string, std::string> thePairT;
      std::string temperatureName = "temperature";
      thePairT = std::make_pair(temperatureName, temperatureName);
      transferVariablesPairName_.push_back(thePairT);
      // T -> Tbc
      std::pair<std::string, std::string> thePairTbc;
      std::string temperatureBcName = "temperature_bc";
      thePairTbc = std::make_pair(temperatureName, temperatureBcName);
      transferVariablesPairName_.push_back(thePairTbc);
    }
    else if ( couplingPhysicsName_ == "thermal_robin" ) {
      // T -> T
      std::pair<std::string, std::string> thePairT;
      std::string temperatureName = "temperature";
      thePairT = std::make_pair(temperatureName, temperatureName);
      transferVariablesPairName_.push_back(thePairT);
      // T -> Tbc
      std::pair<std::string, std::string> thePairTbc;
      std::string temperatureBcName = "temperature_bc";
      thePairTbc = std::make_pair(temperatureName, temperatureBcName);
      transferVariablesPairName_.push_back(thePairTbc);
    }
    else {
      throw std::runtime_error("only supports pre-defined fluids/thermal_cht/robin; perhaps you can use the generic interface");
    }
  }

}

//--------------------------------------------------------------------------
//-------- breadboard ------------------------------------------------------
//--------------------------------------------------------------------------
void
Transfer::breadboard()
{

  // define from and to names
  std::string fromName, toName;

  // realm pair
  fromName = realmPairName_.first;
  toName = realmPairName_.second;

  // extact the realms
  fromRealm_ = root()->realms_->find_realm(fromName);
  if ( NULL == fromRealm_ )
    throw std::runtime_error("from realm in xfer is NULL");
  toRealm_ = root()->realms_->find_realm(toName);
  if ( NULL == toRealm_ )
    throw std::runtime_error("to realm in xfer is NULL");

  // advertise this transfer to realm; for calling control
  fromRealm_->augment_transfer_vector(this, transferObjective_, toRealm_);
 
  // meta data; bulk data to early to extract?
  stk::mesh::MetaData &fromMetaData = fromRealm_->meta_data();
  stk::mesh::MetaData &toMetaData = toRealm_->meta_data();

  // mesh part pairs
  fromName = meshPartPairName_.first;
  toName = meshPartPairName_.second;

  // get the part; no need to subset
  const stk::mesh::Part *fromTargetPart = fromMetaData.get_part(fromName);
  if ( NULL == fromTargetPart )
    throw std::runtime_error("from target part in xfer is NULL");
  const stk::mesh::Part *toTargetPart = toMetaData.get_part(toName);
  if ( NULL == toTargetPart )
    throw std::runtime_error("to target part in xfer is NULL");

  meshPartPair_ = std::make_pair(fromTargetPart, toTargetPart);

  // could extract the fields from the realm now and save them off.... STATE....

  // output
  const bool doOutput = true;
  if ( doOutput ) {

    // realm names
    NaluEnv::self().naluOutputP0() << "Xfer Setup Information: " << name_ << std::endl;
    NaluEnv::self().naluOutputP0() << "the From realm name is: " << fromRealm_->name_ << std::endl;
    NaluEnv::self().naluOutputP0() << "the To realm name is: " << toRealm_->name_ << std::endl;

    // extract mesh part names for the user
    NaluEnv::self().naluOutputP0() << "the From mesh part name is: " << meshPartPair_.first->name() << std::endl;
    NaluEnv::self().naluOutputP0() << "the To mesh part name is: " << meshPartPair_.second->name() << std::endl;
    
    // provide field names
    for( std::vector<std::pair<std::string, std::string> >::const_iterator i_var = transferVariablesPairName_.begin();
	 i_var != transferVariablesPairName_.end(); ++i_var ) {
      const std::pair<std::string, std::string> thePair = *i_var;
      NaluEnv::self().naluOutputP0() << "From variable " << thePair.first << " To variable " << thePair.second << std::endl;
    }
  }
}

void Transfer::allocate_stk_transfer() {

  const stk::mesh::MetaData    &fromMetaData = fromRealm_->meta_data();
        stk::mesh::BulkData    &fromBulkData = fromRealm_->bulk_data();
  const std::string            &fromcoordName   = fromRealm_->get_coordinates_name();
  const std::vector<std::pair<std::string, std::string> > &FromVar = transferVariablesPairName_;
  const stk::ParallelMachine    &fromComm    = fromRealm_->bulk_data().parallel();

  boost::shared_ptr<FromMesh >
    from_mesh (new FromMesh(fromMetaData, fromBulkData, *fromRealm_, fromcoordName, FromVar, meshPartPair_.first, fromComm));

  stk::mesh::MetaData    &toMetaData = toRealm_->meta_data();
  stk::mesh::BulkData    &toBulkData = toRealm_->bulk_data();
  const std::string             &tocoordName   = toRealm_->get_coordinates_name();
  const std::vector<std::pair<std::string, std::string> > &toVar = transferVariablesPairName_;
  const stk::ParallelMachine    &toComm    = toRealm_->bulk_data().parallel();

  boost::shared_ptr<ToMesh >
    to_mesh (new ToMesh(toMetaData, toBulkData, tocoordName, toVar, meshPartPair_.second, toComm));

  typedef stk::transfer::GeometricTransfer< class LinInterp< class FromMesh, class ToMesh > > STKTransfer;

  // extract search type
  stk::search::SearchMethod searchMethod = stk::search::BOOST_RTREE;
  if ( searchMethodName_ == "boost_rtree" )
    searchMethod = stk::search::BOOST_RTREE;
  else if ( searchMethodName_ == "stk_octree" )
    searchMethod = stk::search::OCTREE;
  else
    NaluEnv::self().naluOutputP0() << "Transfer::search method not declared; will use BOOST_RTREE" << std::endl;
  const double expansionFactor = 1.5;
  transfer_.reset(new STKTransfer(from_mesh, to_mesh, name_, expansionFactor, searchMethod));
}

void Transfer::ghost_from_elements()
{
  typedef stk::transfer::GeometricTransfer< class LinInterp< class FromMesh, class ToMesh > > STKTransfer;

  const boost::shared_ptr<STKTransfer> transfer =
      boost::dynamic_pointer_cast<STKTransfer>(transfer_);
  const boost::shared_ptr<STKTransfer::MeshA> mesha = transfer->mesha();

  STKTransfer::MeshA::EntityProcVec entity_keys;
  transfer->determine_entities_to_copy(entity_keys);
  mesha->update_ghosting(entity_keys);
}

//--------------------------------------------------------------------------
//-------- initialize_begin ------------------------------------------------------
//--------------------------------------------------------------------------
void
Transfer::initialize_begin()
{
  NaluEnv::self().naluOutputP0() << "PROCESSING Transfer::initialize_begin() for: " << name_ << std::endl;
  double time = -stk::cpu_time();
  allocate_stk_transfer();
  transfer_->coarse_search();
  time += stk::cpu_time();
  fromRealm_->timerTransferSearch_ += time;
}
//--------------------------------------------------------------------------
//--------------------------------------------------------------------------
void
Transfer::change_ghosting()
{
  ghost_from_elements();
}
//--------------------------------------------------------------------------
//-------- initialize_end ------------------------------------------------------
//--------------------------------------------------------------------------
void
Transfer::initialize_end()
{
  NaluEnv::self().naluOutputP0() << "PROCESSING Transfer::initialize_end() for: " << name_ << std::endl;
  transfer_->local_search();
}

//--------------------------------------------------------------------------
//-------- execute ---------------------------------------------------------
//--------------------------------------------------------------------------
void
Transfer::execute()
{
  // do the xfer
  NaluEnv::self().naluOutputP0() << std::endl;
  NaluEnv::self().naluOutputP0() << "PROCESSING Transfer::execute() for: " << name_ << std::endl;

  // provide field names
  for( std::vector<std::pair<std::string, std::string> >::const_iterator i_var = transferVariablesPairName_.begin();
       i_var != transferVariablesPairName_.end(); ++i_var ) {
    const std::pair<std::string, std::string> thePair = *i_var;
    NaluEnv::self().naluOutputP0() << "XFER From variable: " << thePair.first << " To variable " << thePair.second << std::endl;
  }
  NaluEnv::self().naluOutputP0() << std::endl;
  transfer_->apply();
}

Simulation *Transfer::root() { return parent()->root(); }
Transfers *Transfer::parent() { return &transfers_; }

} // namespace nalu
} // namespace Sierra
