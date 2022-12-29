#include "IvyFramework/IvyMLTools/interface/IvyXGBoostInterface.h"
#include "IvyFramework/IvyDataTools/interface/IvyCSVReader.h"
#include "IvyFramework/IvyDataTools/interface/HelperFunctionsCore.h"
#include "IvyFramework/IvyDataTools/interface/IvyStreamHelpers.hh"


using namespace std;
using namespace IvyStreamHelpers;


void testIvyXGBTrainer(){
  IvyCSVReader csv("test_data_ivyxgb.csv");
  auto fields = csv.getLabels();
  std::vector<TString> coordnames;
  std::vector<TString> prednames;
  for (auto ff:fields){
    if (ff.find("coord:")==0) coordnames.push_back(ff);
    else if (ff.find("pred:")==0) prednames.push_back(ff);
  }

  unsigned long long int nrows = csv.getNRows();
  std::vector<std::vector<IvyMLWrapper::IvyMLDataType_t>> coords; coords.assign(nrows, std::vector<IvyMLWrapper::IvyMLDataType_t>());
  std::vector<std::vector<IvyMLWrapper::IvyMLDataType_t>> preds_csv; preds_csv.assign(nrows, std::vector<IvyMLWrapper::IvyMLDataType_t>());
  for (auto const& var:coordnames){
    auto const& vals = csv.getColumn(var.Data());
    auto it_fillvals = coords.begin();
    for (auto const& val:vals){
      IvyMLWrapper::IvyMLDataType_t vval=0;
      HelperFunctions::castStringToValue(val, vval);
      it_fillvals->push_back(vval);
      it_fillvals++;
    }
  }
  for (auto const& var:prednames){
    auto const& vals = csv.getColumn(var.Data());
    auto it_fillvals = preds_csv.begin();
    for (auto const& val:vals){
      IvyMLWrapper::IvyMLDataType_t vval=0;
      HelperFunctions::castStringToValue(val, vval);
      it_fillvals->push_back(vval);
      it_fillvals++;
    }
  }

  IvyXGBoostInterface xgb;
  xgb.build("test_model_ivyxgb.bin", coordnames, -999.);

  IVYout << "Testing the predicions..." << endl;
  std::vector<std::vector<IvyMLWrapper::IvyMLDataType_t>> preds; preds.assign(nrows, std::vector<IvyMLWrapper::IvyMLDataType_t>());
  {
    auto it_coords = coords.begin();
    auto it_preds_csv = preds_csv.begin();
    auto it_preds = preds.begin();
    for (unsigned long long int irow=0; irow<nrows; irow++){
      HelperFunctions::progressbar(irow, nrows);

      std::unordered_map<TString, IvyMLWrapper::IvyMLDataType_t> tmp_map;
      for (unsigned short ic=0; ic<coordnames.size(); ic++) tmp_map[coordnames.at(ic)] = it_coords->at(ic);
      xgb.eval(tmp_map, *it_preds);

      if (it_preds->size() != it_preds_csv->size()){
        IVYerr << "Size of predictions " << it_preds->size() << " is not " << it_preds_csv->size() << "." << endl;
        break;
      }
      {
        auto it_pred = it_preds->begin();
        auto it_pred_csv = it_preds_csv->begin();
        unsigned short ipred = 0;
        while (it_pred!=it_preds->end()){
          if (std::abs((*it_pred) - (*it_pred_csv))>std::max((*it_pred), (*it_pred_csv))*1e-3){
            IVYerr << "Prediction " << ipred << " for row " << irow << " is significantly different." << endl;
          }
          it_pred++;
          it_pred_csv++;
          ipred++;
        }
      }

      it_coords++;
      it_preds_csv++;
      it_preds++;
    }
  }
}