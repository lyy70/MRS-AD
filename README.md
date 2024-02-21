# MRS-AD
Multi-Scale Rail Surface Anomaly Detection Based on Weighted Multivariate Gaussian distribution
  
Abstract ： Rail surface anomaly detection, referring to the process of identifying and localizing abnormal patterns in rail surface images, faces the limitation of robustness because of the large diversity of scale, quantity, and morphology of surface anomalies.To address this challenge, we propose a multi-scale rail surface anomaly detection method (MRS-AD) based on a distribution model, which cooperates neighborhood information to precisely locate rail surface anomalies. Specifically, MRS-AD integrates multi-scale structures to enhance the perception of different scale information of anomalies.Furthermore, the neighborhood information is utilized to capture the correlations between adjacent regions, and thereby a weighted multivariate Gaussian distribution model is estimated to improve the recognition capability of anomalous morphologies.To validate the effectiveness of MRS-AD, we collected and built a Rail Surface Anomaly Detection dataset (RSAD), considering the scale and quantity of rail surface anomalies. Extensive experiments on RSAD, RSDD and NEU-RSDD-2 demonstrate the superiority of MRS-AD.

# requirements
python == 3.8

pytorch == 1.5

tqdm

sklearn

matplotlib

# Datasets
 RSAD dataset：DownLoad from 链接: https://pan.baidu.com/s/18g-flRE__S0ofVdI3mO_Lw 提取码: RSAD 
 
 RSDD and NEU-RSDD-2 ： https://github.com/neu-rail-rsdds/rail_surface_anomaly_detection
# Result
| method|  MSR-AD(image-wise AUROC) | MSR-AD(Pixel-wise AUROC) | 
| ------------- | ------------- | ------------- |
| mild-disease | 97.3 | 98.2 | 
| moderate-disease  | 97.1  | 95.9 |
| severe-disease  | 97.6 |93.1 | 
| mix-disease  | 97.1 |95.8| 
| average  | 97.3 |95.8| 
