#ifndef IVYXGBOOSTINTERFACE_HPP
#define IVYXGBOOSTINTERFACE_HPP

#include <cassert>
#include "IvyFramework/IvyDataTools/interface/IvyStreamHelpers.hh"
#include "IvyXGBoostInterface.h"


using namespace std;
using namespace IvyStreamHelpers;


#define SAFE_XGBOOST(CALL) \
{ int err_call = (CALL); if (err_call!=0){ IVYerr << "Call '" << #CALL << "' returned error code " << err_call << ". XGBoost last error: " << XGBGetLastError() << endl; } }


template<typename T> bool IvyXGBoostInterface::eval(std::unordered_map<TString, IvyMLDataType_t> const& vars, std::vector<T>& res){
  res.clear();

  constexpr unsigned long long nSample = 1;
  const unsigned long long nFeatures = variable_names.size();
  IvyMLDataType_t* data_arr = new IvyMLDataType_t[nFeatures];
  IvyMLDataType_t* data_arr_ptr = &(data_arr[0]);
  for (auto& vv:variable_names){
    auto it_vars = vars.find(vv);
    if (it_vars==vars.end()) *data_arr_ptr = defval;
    else *data_arr_ptr = it_vars->second;
    data_arr_ptr++;
  };

  bst_ulong nout = 0;
  const float* score;
  DMatrixHandle dvalues;
  SAFE_XGBOOST(XGDMatrixCreateFromMat(data_arr, nSample, nFeatures, defval, &dvalues));
  SAFE_XGBOOST(XGBoosterPredict(*booster, dvalues, 0, 0, &nout, &score));
  SAFE_XGBOOST(XGDMatrixFree(dvalues));

  res.clear();
  res.reserve(nout);
  for (bst_ulong rr=0; rr<nout; rr++) res.push_back(static_cast<T>(score[rr]));

  delete[] data_arr;
  return true;
}

template<typename T> bool IvyXGBoostInterface::eval(std::unordered_map<TString, IvyMLDataType_t> const& vars, T& res){
  std::vector<T> vres;
  bool success = this->eval(vars, vres);
  if (vres.empty() || vres.size()!=1){
    IVYerr << "IvyXGBoostInterface::eval: The vector of results has size = " << vres.size() << " != 1." << endl;
    assert(0);
    success = false;
  }

  res = vres.front();
  return success;
}
template bool IvyXGBoostInterface::eval<float>(std::unordered_map<TString, IvyMLDataType_t> const& vars, float& res);
template bool IvyXGBoostInterface::eval<double>(std::unordered_map<TString, IvyMLDataType_t> const& vars, double& res);

#endif
