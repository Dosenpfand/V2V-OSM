# Code repository for diploma thesis
## Quickstart
1. Install basic tools. On Debian 8

    apt-get install python3 python3-pip git

2. Install libraries. On Debian 8

    apt-get install libfreetype6-dev libxft-dev libgeos-dev libgdal-dev libspatialindex-dev

3. Optionally install linear algebra libraries for a faster numpy experience, on Debian 8

    apt-get install libopenblas-dev liblapack-dev libatlas-dev

4. Clone the repository and cd into it

5. Create a virtual environment

    virtualenv venv

6. Activate the virtual environment

    source venv/bin/activate

7. Install dependencies

    pip install -r requirements.txt

8. Run an exemplary simulation

    python3 main_sim_osm_pathloss.py

9. Get help

    python3 main_sim_osm_pathloss.py -h
