ifndef ROOTSYS
all:
	@echo "ROOTSYS is not set. Please set ROOT environment properly"; echo
else
ifndef ROOFITSYS
all:
	@echo "ROOFITSYS is not set. Please set ROOT environment properly"; echo
else

PROJECTNAME          = IvyFramework
PACKAGENAME          = IvyMLTools

COMPILEPATH          = $(PWD)/
BASEINCLUDE          = $(COMPILEPATH)../../

INCLUDEDIR           = $(COMPILEPATH)interface/
SRCDIR               = $(COMPILEPATH)src/
BINDIR               = $(COMPILEPATH)bin/
SCRIPTSDIR           = $(COMPILEPATH)scripts/
OBJDIR               = $(COMPILEPATH)obj/
LIBDIR               = $(COMPILEPATH)lib/
TESTDIR              = $(COMPILEPATH)test/
RUNDIR               = $(COMPILEPATH)
LIBSHORT             = $(PROJECTNAME)$(PACKAGENAME)
LIB                  = lib$(LIBSHORT).so
LIBRULE              = $(LIBDIR)$(LIB)


# IvyFramework essentials
IVYCOREDIR              = $(BASEINCLUDE)IvyFramework/IvyDataTools/
IVYCORELIBDIR           = $(IVYCOREDIR)lib/
IVYCOREINCDIR           = $(IVYCOREDIR)interface/
IVYCORECXXFLAGS =  -I$(IVYCOREINCDIR) -L$(IVYCORELIBDIR)
IVYCORELIBS =  -lIvyFrameworkIvyDataTools

# XGBOOST essentials
XGBOOSTLIBDIR = ${XGBOOST_PATH}/lib/
XGBOOSTINCDIR = $(XGBOOSTLIBDIR)../include/
RABITINCDIR = $(XGBOOSTLIBDIR)../rabit/include/
XGBOOSTCXXFLAGS = -I$(XGBOOSTINCDIR) -I$(RABITINCDIR) -L$(XGBOOSTLIBDIR)
XGBOOSTLIBS =  -lxgboost


# Here begins compilation...
EXTCXXFLAGS   = $(IVYCORECXXFLAGS) $(XGBOOSTCXXFLAGS)
EXTLIBS       = $(IVYCORELIBS) $(XGBOOSTLIBS)

ROOTCFLAGS    = $(shell root-config --cflags) -Lrootlib
ROOTLIBS     = $(shell root-config --libs) -lMathMore -lGenVector -Lrootlib

ARCH         := $(shell root-config --arch)

CXX           = g++
CXXINC        = -I$(ROOFITSYS)/include/ -I$(BASEINCLUDE) -I$(INCLUDEDIR)
CXXDEFINES    = -D_COMPILE_STANDALONE_
CXXFLAGS      = -fPIC -g -O2 $(ROOTCFLAGS) $(CXXDEFINES) $(CXXINC) $(EXTCXXFLAGS)
LINKERFLAGS   = -Wl,-rpath=$(LIBDIR),-soname,$(LIB)

NLIBS         = $(ROOTLIBS)
# Hack here, because RooFit is removed from ROOT:
NLIBS        += -L $(ROOFITSYS)/lib/ -lMinuit -lRooFitCore -lRooFit #-lboost_regex
# Libraries for common user packages
NLIBS        += $(EXTLIBS)
# Filter out libNew because it leads to floating-point exceptions in versions of ROOT prior to 6.08.02
# See the thread https://root-forum.cern.ch/t/linking-new-library-leads-to-a-floating-point-exception-at-startup/22404
LIBS          = $(filter-out -lNew, $(NLIBS))


SOURCESCC = $(wildcard $(SRCDIR)*.cc)
SOURCESCXX = $(wildcard $(SRCDIR)*.cxx)
OBJECTSPRIM = $(SOURCESCC:.cc=.o) $(SOURCESCXX:.cxx=.o)
OBJECTS = $(subst $(SRCDIR),$(OBJDIR),$(OBJECTSPRIM))
DEPS = $(OBJECTS:.o=.d)

BINSCC = $(wildcard $(BINDIR)*.cc)
BINSCXX = $(wildcard $(BINDIR)*.cxx)


.PHONY: all help compile clean
.SILENT: alldirs clean $(OBJECTS) $(DEPS) $(LIBRULE)


all: $(OBJECTS) $(LIBRULE)


alldirs:
	mkdir -p $(OBJDIR); \
	mkdir -p $(LIBDIR);

$(LIBRULE):	$(OBJECTS) | alldirs
	echo "Linking $(LIB)"; \
	$(CXX) $(LINKERFLAGS) -shared $(OBJECTS) -o $@

$(OBJDIR)%.d:	$(SRCDIR)%.c* | alldirs
	echo "Checking dependencies for $<"; \
	$(CXX) -MM -MT $@ $(CXXFLAGS) $< > $@; \
                     [ -s $@ ] || rm -f $@

$(OBJDIR)%.o: 	$(SRCDIR)%.c* $(OBJDIR)%.d | alldirs
	echo "Compiling $<"; \
	$(CXX) $(CXXFLAGS) $< -c -o $@ $(LIBS)

clean:
	rm -rf $(OBJDIR)
	rm -rf $(LIBDIR)
	rm -f $(SRCDIR)*.o
	rm -f $(SRCDIR)*.so
	rm -f $(SRCDIR)*.d
	rm -f $(BINDIR)*.o
	rm -f $(BINDIR)*.so
	rm -f $(BINDIR)*.d
	rm -rf $(RUNDIR)Pdfdata
	rm -f $(RUNDIR)*.DAT
	rm -f $(RUNDIR)*.dat
	rm -f $(RUNDIR)br.sm*
	rm -f $(RUNDIR)*.cc
	rm -f $(RUNDIR)*.o
	rm -f $(RUNDIR)*.so
	rm -f $(RUNDIR)*.d
	rm -f $(RUNDIR)*.pcm
	rm -f $(RUNDIR)*.pyc
	rm -rf $(TESTDIR)Pdfdata
	rm -f $(TESTDIR)*.DAT
	rm -f $(TESTDIR)*.dat
	rm -f $(TESTDIR)br.sm*
	rm -f $(TESTDIR)*.o
	rm -f $(TESTDIR)*.so
	rm -f $(TESTDIR)*.d
	rm -f $(TESTDIR)*.pcm
	rm -f $(TESTDIR)*.pyc


include $(DEPS)


endif
endif
