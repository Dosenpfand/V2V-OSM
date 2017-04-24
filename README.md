# Code repository for diploma thesis
## Quickstart
Applies to Debian 8

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

10. Get help

    python3 main_sim_osm_pathloss.py -h


## Using SUMO data

[//]: # TODO: more extensive howto!

1. Download and install SUMO from http://sumo.dlr.de
2. Select and export scenario

    python2 /usr/lib/sumo/tools/osmWebWizard.py

4. Change directory to exported data

    cd ~/Sumo/...

3. Simulate and export trace files

    sumo -c osm.sumocfg --fcd-output sumoTrace.xml
