# crane_planning

First-slice ROS-free shell for W05. It installs no C++ headers, libraries, or
runtime nodes; the request/result fixture is private to the package tests and
only exercises the installed `crane_model` mock. Geometric planning, Coal,
OMPL, acados timing, ROS services, the legacy `CalcMovement` adapter, and
runtime nodes are deliberately not implemented here.
