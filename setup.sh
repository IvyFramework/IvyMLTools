#!/bin/bash

(

set -euo pipefail

cd $(dirname ${BASH_SOURCE[0]})

PKGDIR="$(readlink -f .)"
declare -i doPrintEnv=0
declare -i doPrintEnvInstr=0
declare -i needROOFITSYS_ROOTSYS=0
declare -i doClean=0
declare -a setupArgs=()
xgboost_path=""
ivydatatools=""

for farg in "$@"; do
  fargl="$(echo $farg | awk '{print tolower($0)}')"
  if [[ "$fargl" == "env" ]]; then
    doPrintEnv=1
  elif [[ "$fargl" == "envinstr" ]]; then
    doPrintEnvInstr=1
  elif [[ "$fargl" == "xgboost_path="* ]]; then
    xgboost_path="${farg#*=}"
  else
    setupArgs+=( "$farg" )
    if [[ "$farg" == "clean" ]]; then
      doClean=1
    fi
  fi
done
declare -i nSetupArgs
nSetupArgs=${#setupArgs[@]}

printenv() {
  if [[ -d ${ivydatatools} ]]; then
    envopts="env standalone"
    ${ivydatatools}/setup.sh ${envopts}
    eval $(${ivydatatools}/setup.sh ${envopts})
  fi

  if [[ -z "${XGBOOST_PATH+x}" ]]; then
    echo "export XGBOOST_PATH=${xgboost_path}"
    export XGBOOST_PATH=${xgboost_path}
  fi

  libappend="${PKGDIR}/lib"
  end=""
  if [[ ! -z "${LD_LIBRARY_PATH+x}" ]]; then
    end=":${LD_LIBRARY_PATH}"
  fi
  if [[ "${end}" != *"$libappend"* ]]; then
    echo "export LD_LIBRARY_PATH=${libappend}${end}"
    export LD_LIBRARY_PATH=${libappend}${end}
  fi

  libappend="${xgboost_path}/lib"
  end=""
  if [[ ! -z "${LD_LIBRARY_PATH+x}" ]]; then
    end=":${LD_LIBRARY_PATH}"
  fi
  if [[ "${end}" != *"$libappend"* ]]; then
    echo "export LD_LIBRARY_PATH=${libappend}${end}"
    export LD_LIBRARY_PATH=${libappend}${end}
  fi
}
doenv() {
  if [[ -d ${ivydatatools} ]]; then
    envopts="env standalone"
    eval $(${ivydatatools}/setup.sh ${envopts})
  fi

  export XGBOOST_PATH=${xgboost_path}
}
printenvinstr () {
  echo
  echo "to use this repo, you must run:"
  echo
  echo 'eval $(./setup.sh env)'
  echo "or"
  echo 'eval `./setup.sh env`'
  echo
  echo "if you are using a bash-related shell, or you can do"
  echo
  echo './setup.sh env'
  echo
  echo "and change the commands according to your shell in order to do something equivalent to set up the environment variables."
  echo
}


if [[ ${doClean} -ne 1 ]]; then
  if [[ "${xgboost_path}" == "" ]]; then
    echo "You must pass 'xgboost_path=[PATH]'."
    exit 1
  fi
  xgboost_path="$(readlink -f ${xgboost_path})"

  if [[ ! -d ../IvyDataTools ]]; then
    echo "You must have IvyFramework/IvyDataTools checked out."
    exit 1
  fi
  ivydatatools="$(readlink -f ../IvyDataTools)"
fi


if [[ $doPrintEnv -eq 1 ]]; then
    printenv
    exit
elif [[ $doPrintEnvInstr -eq 1 ]]; then
    printenvinstr
    exit
fi


if [[ $nSetupArgs -eq 0 ]]; then
    setupArgs+=( -j 1 )
    nSetupArgs=2
fi

if [[ "$nSetupArgs" -eq 1 ]] && [[ "${setupArgs[0]}" == *"clean"* ]]; then
    make clean

    exit $?
elif [[ "$nSetupArgs" -ge 1 ]] && [[ "$nSetupArgs" -le 2 ]] && [[ "${setupArgs[0]}" == *"-j"* ]]; then
    : ok
else
    echo "Unknown arguments:"
    echo "  ${setupArgs[@]}"
    echo "Should be nothing, env, envinstr, clean, or -j [Ncores]"
    exit 1
fi


doenv


# Run arguments for IvyDataTools
${ivydatatools}/setup.sh standalone "${setupArgs[@]}" 1> /dev/null || exit $?

# Compile this repository
make "${setupArgs[@]}"
compile_status=$?
if [[ ${compile_status} -ne 0 ]]; then
  echo "Compilation failed with status ${compile_status}."
  exit ${compile_status}
fi

printenvinstr

)
