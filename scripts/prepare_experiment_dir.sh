#!/bin/bash
# Prepare given experiment directory by creating all directories
# and populating it with common information, e.g: current git branch and commit,
# current diff.

mkdir -p ${1}

# Store current branch and commit
git show -s > ${1}/git_commit.txt
# Store current diff
git diff --output=${1}/git_diff.txt 
