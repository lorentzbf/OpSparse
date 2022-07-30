#!/bin/bash

if [ -d matrix/suite_sparse ]; then
    cd matrix/suite_sparse
else
    mkdir -p matrix/suite_sparse
    cd matrix/suite_sparse
fi

# download webbase-1M
if [ ! -e webbase-1M/webbase-1M.mtx ]; then
    wget https://suitesparse-collection-website.herokuapp.com/MM/Williams/webbase-1M.tar.gz
    tar zxvf webbase-1M.tar.gz
fi
echo Successfully downloaded the matrix.

