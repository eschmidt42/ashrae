#!/bin/bash
cwd=$(pwd)
echo 'Linking ashrae/ and nbs/'
ln -s "${cwd}"/ashrae/ "${cwd}"/nbs/ashrae
echo 'Linking ashrae/ and other_nbs/'
ln -s "${cwd}"/ashrae/ "${cwd}"/other_nbs/ashrae
