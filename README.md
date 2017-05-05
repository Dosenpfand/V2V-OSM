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

10. Modify the simulation parameters in

        network_definition.json

    and run the simulation again

        python3 main_sim_osm_pathloss.py

11. Optional: For vehicle routing and movement install SUMO ( www.sumo.dlr.de )

        apt-get install sumo
