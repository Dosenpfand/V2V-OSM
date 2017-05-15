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

8. Install dependencies

        pip install -r requirements.txt

9. Run an exemplary simulation

        python3 main_sim_osm_pathloss.py

10. Modify the simulation parameters in

        network_definition.json

    and run the simulation again

        python3 main_sim_osm_pathloss.py

11. Optional: For vehicle realistic vehicle placement and movement install [SUMO](http://www.sumo.dlr.de)

        apt-get install sumo

# Tests
## Unit tests
To run the included unit tests execute

    python3 -m unittest discover

## Coverage
To determine the coverage of the unit tests install dependencies with
    
    pip install -r requirements-test.txt

Then run the tests again with coverage analysis

    coverage run --source=. -m unittest discover

Create a HTML report with

    coverage html

And then open `htmlcov/index.html`.

To create a badge execute

    coverage-badge -f -o .travis/coverage.svg
