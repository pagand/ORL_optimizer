Here is the Simulator and suggested display for the Human-in-the-loop optimization. 
The only data needed for this is in the data/VesselSimulator folder
Create a seperate venv for this simulator, given the simulatorrequirements.txt


```bash
pip install virtualenv
virtualenv -p /path/to/python3.11 .venv
source .venv/bin/activate
pip install -r VesselSimulator/simulatorrequirements.txt
```


Install torch locally given this (URL)[https://pytorch.org/get-started/locally/].
