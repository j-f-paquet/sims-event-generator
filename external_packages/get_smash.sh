#!/usr/bin/env bash

###############################################################################
# Copyright (c) The JETSCAPE Collaboration, 2018
#
# For the list of contributors see AUTHORS.
#
# Report issues at https://github.com/JETSCAPE/JETSCAPE/issues
#
# or via email to bugs.jetscape@gmail.com
#
# Distributed under the GNU General Public License 3.0 (GPLv3 or later).
# See COPYING for details.
##############################################################################

# 1) Download the SMASH code
git clone https://github.com/smash-transport/smash.git smash/smash_code

# 2) Compile SMASH
cd smash/smash_code
#checkout the commit version that was used for production runs 
git checkout 0063efcc88c11151fa4422940a8bd145a52c356d

#update March 7, 2019 Smash Bug fixed w.r.t. formation time
#if necessary for future stability, checkout a fixed commit after this date
rm -r build
mkdir build
cd build
cmake .. -DPythia_CONFIG_EXECUTABLE=${PYTHIA8DIR}/bin/pythia8-config
number_of_cores=`nproc --all`
number_of_cores_to_compile=$(( ${number_of_cores} > 20 ? 20 : ${number_of_cores} ))
echo "Compiling SMASH using ${number_of_cores_to_compile} cores."
make -j${number_of_cores_to_compile} smash_shared


#Modification I had to make to SMASH to get it to compile
#diff --git a/src/include/smash/fourvector.h b/src/include/smash/fourvector.h
#index 660022f4..333752f0 100644
#--- a/src/include/smash/fourvector.h
#+++ b/src/include/smash/fourvector.h
#@@ -10,6 +10,7 @@
# #include <array>
# #include <cmath>
# #include <iosfwd>
#+#include <stdexcept>
#
# #include "threevector.h"
#
#diff --git a/src/include/smash/logging.h b/src/include/smash/logging.h
#index 73c991ce..4015eed6 100644
#--- a/src/include/smash/logging.h
#+++ b/src/include/smash/logging.h
#@@ -12,6 +12,8 @@
#
# #include <stdexcept>
# #include <tuple>
#+#include <boost/throw_exception.hpp>
#+#include <boost/exception/diagnostic_information.hpp>
#
# #include <yaml-cpp/yaml.h>  // NOLINT(build/include_order)
# #include <einhard.hpp>
#diff --git a/src/include/smash/threevector.h b/src/include/smash/threevector.h
#index 4585110e..5a8683fe 100644
#--- a/src/include/smash/threevector.h
#+++ b/src/include/smash/threevector.h
#@@ -12,6 +12,8 @@
#
# #include <array>
# #include <cmath>
#+#include <ostream>
#+
#
# #include "constants.h"

