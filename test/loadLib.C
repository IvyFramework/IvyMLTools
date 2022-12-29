{
  TString this_file = __FILE__;
  TString this_dir = this_file(0, this_file.Last('/'));
  if (!this_dir.BeginsWith(".") && this_dir.EndsWith(".")) this_dir = this_dir(0, this_dir.Last('/'));
  this_dir = this_dir(0, this_dir.Last('/'));
  this_dir = this_dir(0, this_dir.Last('/'));
  this_dir = this_dir(0, this_dir.Last('/'));

  TString inc_this_dir = Form("-I%s", this_dir.Data());
  gSystem->AddIncludePath(inc_this_dir+"/IvyFramework/IvyDataTools/interface/");
  gSystem->AddIncludePath(inc_this_dir+"/IvyFramework/IvyMLTools/interface/");
  gSystem->AddIncludePath(inc_this_dir);

  gSystem->Load("libCore");
  gSystem->Load("libRIO");
  gSystem->Load("libMatrix");
  gSystem->Load("libMinuit");
  gSystem->Load("libMathCore");
  gSystem->Load("libMathMore");
  gSystem->Load("libPhysics");
  gSystem->Load("libHist");
  gSystem->Load("libTree");
  gSystem->Load("libGraf");
  gSystem->Load("libRooFitCore");
  gSystem->Load("libRooFit");
  gSystem->Load("libIvyFrameworkIvyDataTools.so");
  gSystem->Load("libIvyFrameworkIvyMLTools.so");
}
