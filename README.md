Here is the file to reproduce the results of our analysis of three Vanilla RNNs via Spectral-Submanifold (SSM) model reduction:
- The "Context Dependent Decision-Making RNN" of Mante et al. (2013): the analysis is in "cddm demo.ipynb"; 
- The "Sine-Wave Generator RNN" of Krause et al. (2022): the analysis is in "swg demo.ipynb";
- The "Multitasking RNN" of Driscoll et al. (2024) performing a "Memory-Pro task": the analysis is in "mt demo.ipynb";

SSM reduction was performed, given simulated test and train trajectories, through the Matlab package SSMLearn (https://github.com/haller-group/SSMLearn); results were saved in the "Data" subfolder of each experiment.
Helper functions for constructing the SSM parametrization and reduced dynamics in Python can be found in the helper file "SSMfunctions.py".


