# ULTra-AV: A Unified Longitudinal Trajectory dataset for Automated Vehicle

## Introduction

This repo provides the source code and data for the following paper:

ULTra-AV: A Unified Longitudinal Trajectory Dataset for Automated Vehicle

In this paper, we processed a unified trajectory dataset for automated vehicles'  longitudinal behavior from 13 distinct sources. The extraction and cleaning of the dataset contains the following three steps -  1. extraction of longitudinal trajectory data, 2. general data cleaning, and 3. data-specific cleaning. The dataset obtained from step 2 and step 3 are named as the longitudinal trajectory data and car-following trajectory data. We also analyzed and validated the data by multiple methods. The obtained datasets and the python code used to analyzed the datasets are all provided in this repo. We hope this dataset can benefit the study of  microscopic longitudinal AV behaviors.

## Original Datasets

We have examined 13 open-source datasets, each providing distinct insights into AV behavior across various driving conditions and scenarios. These open-source datasets are from six providers:

- **Vanderbilt ACC Dataset** [1]. Collected in Nashville, Tennessee by Vanderbilt University research group. Available at - [https://acc-dataset.github.io/datasets/](https://acc-dataset.github.io/datasets/).
  - [Two-vehicle ACC driving, Tennessee 2019](https://github.com/CATS-Lab/Filed-Experiment-Data-AV_Platooning_Data)
- **CATS Open Datasets** [2]. Three datasets were gathered in Tampa, Florida, and Madison, Wisconsin by the CATS Lab. Available at - [https://github.com/CATS-Lab](https://github.com/CATS-Lab).
  - [Filed-Experiment-Data-AV_Platooning_Data](https://github.com/CATS-Lab/Filed-Experiment-Data-AV_Platooning_Data)
  - [Filed-Experiment-Data-ACC_Data](https://github.com/CATS-Lab/Filed-Experiment-Data-ACC_Data)
- **OpenACC Database** [3]. Four datasets were collected across Italy, Sweden, and Hungary by the European Commission's Joint Research Centre. Available at - [https://data.europa.eu/data/datasets/9702c950-c80f-4d2f-982f-44d06ea0009f?locale=en](https://data.europa.eu/data/datasets/9702c950-c80f-4d2f-982f-44d06ea0009f?locale=en).
  - [Casale](https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/TransportExpData/JRCDBT0001/LATEST/Casale/)
  - [Vicolungo](https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/TransportExpData/JRCDBT0001/LATEST/Vicolungo/)
  - [AstaZero](https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/TransportExpData/JRCDBT0001/LATEST/AstaZero/)
  - [ZalaZone](https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/TransportExpData/JRCDBT0001/LATEST/ZalaZone/)
- **Central Ohio ACC Datasets** [4]. Two datasets were collated in Ohio by UCLA's Mobility Lab and Transportation Research Center. Available at - 
  - [Advanced Driver Assistance System (ADAS)-Equipped Single-Vehicle Data for Central Ohio](https://catalog.data.gov/dataset/advanced-driver-assistance-system-adas-equipped-single-vehicle-data-for-central-ohio)
  - [Advanced Driver Assistance System (ADAS)-Equipped Two-Vehicle Data for Central Ohio](https://catalog.data.gov/dataset/advanced-driver-assistance-system-adas-equipped-two-vehicle-data-for-central-ohio)
- **Waymo Open Dataset** [5, 6]. Two datasets were collected in six cities including San Francisco, Mountain View, and Los Angeles in California, Phoenix in Arizona, Detroit in Michigan, and Seattle in Washington by Waymo. Available at - 
  - [Waymo Motion Dataset](https://waymo.com/open/data/motion/)
  - [Vehicle trajectory data processed from the Waymo Open Dataset](https://data.mendeley.com/datasets/wfn2c3437n/2)
- **Argoverse 2 Motion Forecasting Dataset** [7]. Collected from Austin in Texas, Detroit in Michigan, Miami in Florida, Pittsburgh in Pennsylvania, Palo Alto in California, and Washington, D.C. by Argo AI with researchers from Carnegie Mellon University and the Georgia Institute of Technology. Available at - 
  - [Argoverse 2 Motion Forcasting Dataset]([Argoverse 2](https://www.argoverse.org/av2.html))

For more details of the datasets, please refer to the reference and our paper.

## Installation

All the data are provided by CSV format. If you want to run our source code, please make sure you follow the prerequisites below:

1. **Python 3** - Ensure you have a Python 3 environment set up.
2. **Required Packages** - Install all necessary packages listed in the `requirements.txt` file.
3. **Original data or the processed dataset** - Download the original data or our processed data from the link we provided and check the path of the data with our code if you want to process or analyze the data.

We also recommend using other software packages such as R to effectively analyze the trajectory data. These tools are well-suited for handling the dataset's format.

## Usage

### Code

The code related to our data processing and validation are all stored in folder `\Code`. This folder contain the following files:

- **Main.py** - The main function calls data processing and analysis functions for each dataset.
- **trajectory_extraction.py** - Code used in Step 1 to extract AV longitudinal trajectories.
- **data_transformation.py** - Code used in Step 1 to convert all datasets to a unified format.
- **data_cleaning.py** - Code used in Steps 2 and 3 for data cleaning.
- **data_analysis.py** - Code used to analyze data statistics, plot traffic performance of datasets, and plot scatter plots.
- **model_calibration.py** - An example tool to use the processed data to calibrate a linear car-following model.

To use this repo, run Python script `Main.py`. As you proceed through each Python script, always verify the paths for both the input and output files. This ensures that everything runs smoothly.

### Data

Extracted datasets are stored in [Box link](https://uwmadison.box.com/s/6e74ehpj2rx6kbjnma1b1ok4fupe5nkb).

Data attributes are shown below:

| Label         | Description                                  | Notations and formulation                                    | Unit |
| ------------- | -------------------------------------------- | ------------------------------------------------------------ | ---- |
| Trajectory_ID | ID of the longitudinal trajectory.           | $i\in \mathcal{I}$.                                          | N/A  |
| Time_Index    | Common time stamp in one trajectory.         | $t\in \mathcal{T}_i, i\in \mathcal{I}$.                      | s    |
| ID_LV         | LV (lead vehicle) ID.                        | $c^{\mathrm{l}}_i, i\in \mathcal{I}$. Label each FAV with a different ID and all HVs with -1. | N/A  |
| Pos_LV        | LV position in the Frenet coordinate.        | $p^{\mathrm{l}}_{it}=p^{\mathrm{f}}_{it}+h_{it}, i\in \mathcal{I}, t\in \mathcal{T}_i$. | m    |
| Speed_LV      | LV speed.                                    | $v^{\mathrm{l}}_{it}=\frac{p^{\mathrm{l}}_{i(t+1)}-p^{\mathrm{l}}_{it}}{\Delta t}, i\in \mathcal{I}, t\in \mathcal{T}_i$. | m/s  |
| Acc_LV        | LV acceleration.                             | $a^{\mathrm{l}}_{it}=\frac{v^{\mathrm{l}}_{i(t+1)}-v^{\mathrm{l}}_{it}}{\Delta t}, i\in \mathcal{I}, t\in \mathcal{T}_i$. | m/s² |
| ID_FAV        | FAV (following automated vehicle) ID.        | $c^{\mathrm{f}}_i, i\in \mathcal{I}$. Label each FAV with a different ID. | N/A  |
| Pos_FAV       | FAV position in the Frenet coordinate.       | $p^{\mathrm{f}}_{it}=p^{\mathrm{f}}_{i(t-1)}+\Delta t \cdot v^{\mathrm{f}}_{it}, i\in \mathcal{I}, t\in \mathcal{T}_i$. | m    |
| Speed_FAV     | FAV speed.                                   | $v^{\mathrm{f}}_{it}=\frac{p^{\mathrm{f}}_{i(t+1)}-p^{\mathrm{f}}_{it}}{\Delta t}, i\in \mathcal{I}, t\in \mathcal{T}_i$. | m/s  |
| Acc_FAV       | FAV acceleration.                            | $a^{\mathrm{f}}_{it}=\frac{v^{\mathrm{f}}_{i(t+1)}-v^{\mathrm{f}}_{it}}{\Delta t}, i\in \mathcal{I}, t\in \mathcal{T}_i$. | m/s² |
| Space_Gap     | Bump-to-bump distance between two vehicles.  | $g_{it}=p^{\mathrm{l}}_{it}-p^{\mathrm{f}}_{it} - l^{\mathrm{f}}/2 -l^{\mathrm{l}}/2, i\in \mathcal{I}, t\in \mathcal{T}_i$, where $l^{\mathrm{f}}$ and $l^{\mathrm{f}}$ are the length of the LV and the FAV. | m    |
| Space_Headway | Distance between the center of two vehicles. | $h_{it}=p^{\mathrm{l}}_{it}-p^{\mathrm{f}}_{it}, i\in \mathcal{I}, t\in \mathcal{T}_i$. | m    |
| Speed_Diff    | Speed difference of the two vehicles.        | $\Delta v_{it}=v^{\mathrm{l}}_{it}-v^{\mathrm{f}}_{it}, i\in \mathcal{I}, t\in \mathcal{T}_i$. | m/s  |

The FAV ID are provided below:

**Vanderbilt Two-vehicle ACC Dataset:**

- 0 - A commercially available 2019 SUV with a full speed range adaptive cruise control system.

**CATS ACC Dataset:**

- 0 - Lincoln MKZs 2016 (Black)
- 1 - Lincoln MKZs 2017 (Red)

**CATS Platoon Dataset:**

- 0 - Lincoln MKZs 2016 (Black)
- 1 - Lincoln MKZs 2017 (Red)

**CATS UW Dataset:**

- 0 - Lincoln MKZs 2017 (Red)

**OpenACC Casale Dataset:**

- 0 - Rexton
- 1 - Hyundai	

**OpenACC Vicolungo Dataset:**

- 0 - Ford(S-Max)
- 1 - KIA(Niro)
- 2 - Mini(Cooper)
- 3 - Mitsubishi(OutlanderPHEV)
- 4 - Mitsubishi(SpaceStar)
- 5 - Peugeot(3008GTLine)
- 6 - VW(GolfE)

**OpenACC Asta Dataset:**

- 0 - Audi(A6)
- 1 - Audi(A8)
- 2 - BMW(X5)
- 3 - Mercedes(AClass)	
- 4 - Tesla(Model3)	

**OpenACC ZalaZone Dataset:**

- 0 - AUDI_A4
- 1 - AUDI_E_TRON
- 2 - BMW_I3
- 3 - JAGUAR_I_PACE
- 4 - MAZDA_3
- 5 - MERCEDES_GLE450
- 6 - SMART_TARGET
- 7 - SKODA_TARGET
- 8 - TESLA_MODEL3
- 9 - TESLA_MODELS
- 10 - TESLA_MODELX
- 11 - TOYOTA_RAV4

**Ohio Single-vehicle Dataset:**

- 0 - retrofitted Tesla Sedan

**Ohio Two-vehicle Dataset:**

- 0 - retrofitted Tesla Sedan
- 1 - retrofitted Ford Fusion Sedan

**Waymo Perception Dataset:**

- 0 - Waymo ADS-equipped vehicle

**Waymo Motion Dataset:**

- 0 - Waymo ADS-equipped vehicle

**Argoverse 2 Motion Forecasting Dataset:**

- 0 - Argo AI self-driving Ford

For more details of the labels and the vehicle types, please refer to our paper.

## Developers

Developer - Hang Zhou (hzhou364@wisc.edu).

Code reviewer - Ke Ma (kma62@wisc.edu).

If you have any questions, please feel free to contact CATS Lab in UW-Madison. We're here to help!

## Reference

[1] Wang, Yanbing, George Gunter, Matthew Nice, and Daniel B. Work. "Estimating adaptive cruise control model parameters from on-board radar units." *arXiv preprint arXiv:1911.06454* (2019).

[2] Shi, Xiaowei, and Xiaopeng Li. "Empirical study on car-following characteristics of commercial automated vehicles with different headway settings." *Transportation research part C: emerging technologies* 128 (2021): 103134.

[3] Makridis, Michail, Konstantinos Mattas, Aikaterini Anesiadou, and Biagio Ciuffo. "OpenACC. An open database of car-following experiments to study the properties of commercial ACC systems." *Transportation research part C: emerging technologies* 125 (2021): 103047.

[4] Xia, Xin, Zonglin Meng, Xu Han, Hanzhao Li, Takahiro Tsukiji, Runsheng Xu, Zhaoliang Zheng, and Jiaqi Ma. "An automated driving systems data acquisition and analytics platform." *Transportation research part C: emerging technologies* 151 (2023): 104120.

[5] Hu, Xiangwang, Zuduo Zheng, Danjue Chen, Xi Zhang, and Jian Sun. "Processing, assessing, and enhancing the Waymo autonomous vehicle open dataset for driving behavior research." *Transportation Research Part C: Emerging Technologies* 134 (2022): 103490.

[6] Ettinger, Scott, Shuyang Cheng, Benjamin Caine, Chenxi Liu, Hang Zhao, Sabeek Pradhan, Yuning Chai et al. "Large scale interactive motion forecasting for autonomous driving: The waymo open motion dataset." In *Proceedings of the IEEE/CVF International Conference on Computer Vision*, pp. 9710-9719. 2021.

[7] Wilson, Benjamin, William Qi, Tanmay Agarwal, John Lambert, Jagjeet Singh, Siddhesh Khandelwal, Bowen Pan et al. "Argoverse 2: Next generation datasets for self-driving perception and forecasting." *arXiv preprint arXiv:2301.00493* (2023).
