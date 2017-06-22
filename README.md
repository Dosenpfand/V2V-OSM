# V2V-OSM
Simulate Vehicle-to-vehicle (V2V) communication on street networks obtained from [OpenStreetMap](https://www.openstreetmap.org/) (OSM)

## Status
[![Build Status](https://travis-ci.com/Dosenpfand/thesis_code.svg?token=q9NYsPfK37J7qYiKq4xe&branch=master)](https://travis-ci.com/Dosenpfand/thesis_code)
![Coverage](https://github.com/Dosenpfand/thesis_code/blob/travis/.travis/coverage.png?raw=true)

## Main Components
Main software components are:

- This Python package
- [OSMnx](https://github.com/gboeing/osmnx)
- [SUMO](http://www.sumo.dlr.de)
- [NetworkX](https://networkx.github.io/)
- To see all third party libraries check `requirements.txt`

## Quickstart
To get started on Debian 8 follow these steps.

1. Install basic tools

        apt-get install python3 python3-pip git

2. Install libraries

        apt-get install libfreetype6-dev libxft-dev libgeos-dev libgdal-dev libspatialindex-dev

3. Optionally install linear algebra libraries for a faster numpy experience

        apt-get install libopenblas-dev liblapack-dev gfortran

    or if you want to use ATLAS instead of OpenBLAS:

        apt-get install liblapack-dev libatlas-dev libatlas-base-dev gfortran

4. Clone the repository and cd into it

5. Create a virtual environment

        python3 -m venv venv --without-pip

6. Activate the virtual environment

        source venv/bin/activate

7. Download and install pip

        wget https://bootstrap.pypa.io/get-pip.py
        python 3 get-pip.py

8. Optional: Install the package

        pip install .
        
   or via symlinks
        
        pip install -e .

9. Run an exemplary simulation

        python3 -m vtovosm.simulations.main

10. Modify the simulation parameters in

        vtovosm/simulations/network_config/default.json

    and run the simulation again

        python3 -m vtovosm.simulations.main
        
11. Get help by executing

        python3 -m vtovosm.simulations.main -h

12. Optional: For realistic vehicle placement and movement install [SUMO](http://www.sumo.dlr.de) from the [backports repository](https://backports.debian.org/Instructions/):

        apt-get -t jessie-backports install sumo sumo-tools

# Tests
To run the tests install the test specific dependencies by executing

    pip install -r requirements-test.txt

## Unit tests


Run the unit tests by executing

    nosetests -v
    
For all tests to complete successfully SUMO needs to be installed (see Quickstart).

To only run fast tests or tests that do not need network access exectute

    nosetests -v -a '!slow'
    
or

    nosetests -v -a '!network'

## Coverage
Run the tests with coverage analysis by starting
    
    nosetests --with-coverage --cover-html --cover-tests --cover-package=vtovosm --verbose

And then open `cover/index.html`.

To create a badge execute

    coverage-badge -f -o .travis/coverage.svg
    
# Authors

- [Markus Gasser](https://github.com/Dosenpfand)
- [Thomas Blazek](https://github.com/tmblazek)
