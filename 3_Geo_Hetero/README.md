# Week Three: Spatial Heterogeneity
So far, to make the problem simpler, our SEIRD model is based on the assumption that the population is geographically homogeneous, i.e. population density and transmission rate are the same across the entire region. However, this is not necessarily true in reality. In this week, we introduce spatial heterogeneity to the model and see how it affects the spread of the disease. Main resources:
- [EpiGeoPop](https://arxiv.org/abs/2310.13468)
- [Assessing the performance of compartmental models in spatially heterogeneous geographies](https://arxiv.org/pdf/2503.04648)

## 3.1 EpiGeoPop
EpiGeoPop is a user-friendly tool for generating **population configurations** (visualizing as well) and **parameters related to age distribution** based on global population data, facilitating and standardizing the complex and time-comsuming model set up in **Agent-based models** (ABMs). ABMs are an alternative to traditional mathematical models, as they can capture spatial heterogeneity, particularly when assessing intervention strategies. Combined with Epiabm, they can demonstrate how spacial patterns influence the spread of disease and the response to intervention.

## 3.2 Assessing compartmental models' performance under spacial heterogeneity
Now we use the ABM parameters and simulated R_t as the ground-truth to assess inference methods for R_t.

According to 3.1, with the aid of EpiGeoPop, we can use ABMs to simulate the spread of disease in a spatially heterogeneous environment. The parameters in ABMs include $\kappa$ and $\gamma$, which can be used for validation later. Then, we can fit SEIR model to the aggregated simulation result, getting the estimated parameters as well as R_t. By comparing the real (ABMs) and estimated (SEIR) parameters and R_t, we can evaluate the performance of the inference methods under spacial heterogeneity.