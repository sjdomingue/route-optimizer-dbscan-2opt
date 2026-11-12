\# Route Optimizer (DBSCAN + Nearest Neighbor + 2-opt)



A Python-based route optimization tool that clusters delivery points using 

DBSCAN, creates an initial route with the Nearest Neighbor algorithm, and 

improves it using the 2-opt heuristic.



\## Problem



Planning efficient delivery or inspection routes for hundreds of scattered 

points was done manually. This resulted in overlapping routes, excessive 

travel time, and high fuel costs. A typical manual planning session for 200 

points took \*\*\~3 hours\*\*.



\## Solution



A script that:

\- Reads a set of points (CSV with X/Y coordinates).

\- Uses \*\*DBSCAN\*\* to group nearby points into zones.

\- Generates an initial route per zone using the \*\*Nearest Neighbor\*\* algorithm.

\- Optimizes the route by eliminating crossing paths using the \*\*2-opt\*\* heuristic.

\- Outputs the optimized route as a CSV ready for mapping or field use.



\## Result



Route planning time reduced from \*\*\~3 hours to 15 minutes\*\*, while also 

reducing total travel distance by \*\*\~20%\*\* compared to the manual approach.



\## Demo



!\[Demo](docs/demo.png)



\## Installation (3 steps)



1\. Clone the repository and install dependencies: `pip install -r requirements.txt`.

2\. Place your input points in `data/sample\_input.csv` (or use the included synthetic data).

3\. Run the script: `python route\_optimizer.py`.



\## Stack



\- Python 3

\- Pandas \& NumPy (data manipulation)

\- Scikit-learn (DBSCAN clustering)

\- Pyproj (coordinate transformation)

\- Synthetic test data included



\## License



MIT — see \[LICENSE](LICENSE).



> Data disclaimer: this repository uses ONLY synthetic data. No real 

> customer locations or proprietary corporate data are included.

