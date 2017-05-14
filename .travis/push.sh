#!/bin/sh

setup_git() {
  git config --global user.email "travis@travis-ci.org"
  git config --global user.name "Travis CI"
}

commit_coverage_image() {
  git checkout -b travis
  git add .travis/coverage.svg
  git commit --message "Travis coverage update: $TRAVIS_BUILD_NUMBER"
}

push_files() {
  git remote add origin-travis https://${GH_TOKEN}@github.com:Dosenpfand/thesis_code.git > /dev/null 2>&1
  git push --quiet --set-upstream origin-travis travis
}

setup_git
commit_coverage_image
push_files
