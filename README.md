# Quantum Federated Learning Experiments in the Cloud
## RESEARCH FOCUS
Quantum Federated Learning (QFL) is an emerging concept that aims to unfold federated learning (FL) over quantum networks, enabling collaborative quantum model training along with local data privacy. We explore the challenges of deploying QFL on cloud platforms, emphasizing quantum intricacies and platform limitations. The proposed data-encoding-driven QFL, with a proof of concept (GitHub Open Source) using genomic data sets on quantum simulators, shows promising results. Investigatros are from IoT \& SE Lab, School of IT, Deakin University, shiva.pokhrel@deakin.edu.au. 

The novelty of this research lies in its potential to expedite QFL adoption, paving the way for quantum-enhanced machine learning models over the cloud trained efficiently in a distributed setting while upholding local privacy. We design a novel process, as illustrated with Qiskit components in the Figure below, that can be perceived as approximately transforming input data into a quantum state, exploring and exploiting it using a customizable parameterized quantum circuit, and iteratively optimizing the parameters to steer and achieve the desired outcome based on the global objective function. 
The technical report and details are discussed in .

In the proposed QFL realization, clients transform their unique data into quantum states using a Feature Map, then process them with a parameterized quantum circuit (Ansatz) where local training is conducted using Qiskit and the updated weights are aggregated centrally, and global weights are returned to clients for local model updates.

![fig3](https://github.com/ShivaPokhrel/qiskit_qfl/blob/main/quant_1.png?raw=true)

##### A high level view of local learning in the proposed QFL Process consisting of several key components. The Feature Map ingests input data and encodes it into a quantum state. Following this, the Ansatz comes into play as a parameterized quantum circuit, with its parameters being iteratively fed by the Optimizer--optimization objective function is driven by the outcomes from the Sampler.

