#!/bin/sh

convert_image() {
  convert .travis/coverage.svg .travis/coverage.png
}

setup_git() {
  git config --global user.email "travis@travis-ci.org"
  git config --global user.name "Travis CI"
}

commit_coverage_image() {
  git checkout -b travis
  git add .travis/coverage.svg .travis/coverage.png
  git commit --message "Travis coverage update: $TRAVIS_BUILD_NUMBER"
}

push_files() {
  git remote add origin-travis https://${GH_TOKEN}@github.com/Dosenpfand/V2V-OSM.git > /dev/null 2>&1
  git push -f --quiet --set-upstream origin-travis travis
}

setup_git
convert_image
commit_coverage_image
push_files
