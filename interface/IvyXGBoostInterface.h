#ifndef IVYXGBOOSTINTERFACE_H
#define IVYXGBOOSTINTERFACE_H

#include <xgboost/c_api.h>
#include "IvyMLWrapper.h"


class IvyXGBoostInterface : public IvyMLWrapper{
protected:
  BoosterHandle* booster;
  IvyMLDataType_t defval;
  std::vector<TString> variable_names;

public:
  IvyXGBoostInterface();
  virtual ~IvyXGBoostInterface();

  bool build(TString fname, std::vector<TString> const& varnames, IvyMLDataType_t missing_entry_val);

  std::vector<TString> const& getVariableNames() const{ return variable_names; }

  BoosterHandle* const& getBooster() const{ return booster; }

  template<typename T> bool eval(std::unordered_map<TString, IvyMLDataType_t> const& vars, std::vector<T>& res);
  template<typename T> bool eval(std::unordered_map<TString, IvyMLDataType_t> const& vars, T& res);

};


#endif
