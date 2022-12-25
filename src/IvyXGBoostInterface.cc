#include <limits>
#include "IvyFramework/IvyDataTools/interface/HostHelpersCore.h"
#include "IvyXGBoostInterface.hpp"


IvyXGBoostInterface::IvyXGBoostInterface() :
  IvyMLWrapper(),
  booster(nullptr),
  defval(0)
{}

IvyXGBoostInterface::~IvyXGBoostInterface(){
  SAFE_XGBOOST(XGBoosterFree(*booster));
  delete booster;
}

bool IvyXGBoostInterface::build(TString fname, std::vector<TString> const& varnames, IvyMLDataType_t missing_entry_val){
  if (booster){
    IVYerr << "IvyXGBoostInterface::build: The booster is already built." << endl;
    return false;
  }
  if (fname == ""){
    IVYerr << "IvyXGBoostInterface::build: The file name is an empty string. This function should be called to load models from a file." << endl;
    assert(0);
  }

  HostHelpers::ExpandEnvironmentVariables(fname);
  if (!HostHelpers::FileExists(fname)){
    IVYerr << "IvyXGBoostInterface::build: File " << fname << " does not exist." << endl;
    assert(0);
  }

  defval = missing_entry_val;
  variable_names = varnames;

  booster = new BoosterHandle;
  SAFE_XGBOOST(XGBoosterCreate(nullptr, 0, booster));

  IVYout << "IvyXGBoostInterface::build: A new xgboost booster is created. Loading the model in " << fname << "..." << endl;

  SAFE_XGBOOST(XGBoosterLoadModel(*booster, fname.Data()));

  return true;
}
