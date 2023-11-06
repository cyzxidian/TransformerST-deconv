# TransformerST-deconv
**Integrating Histology Imaging for Reference-Free Cell-Type Deconvolution in Spatial Transcriptomics**

Our approach presents a cutting-edge reference-free deconvolution technique that elevates spatial transcriptomics analysis by integrating histology images, which have been underutilized by existing methods. 

Framework

![image](https://github.com/cyzxidian/TransformerST-deconv/blob/main/TransformerST-deconv.png) 

The code is licensed under the MIT license. 

**1. Requirements**

**1.1 Operating systems:**

The code in python has been tested on Linux (Ubuntu 20.04.1 LTS).  

**1.2 Required packages in python:**

anndata   
numpy  
opencv-python   
pandas  
python-louvain  
rpy2  
scanpy  
scipy  
seaborn   
torch  
torch-geometric    
torchvision  
tqdm  
umap-learn  

**1.3 How to install TransformerST-deconv:**  
(1) cd TransformerST-deconv  
(2) conda create --name TransformerST-deconv  
(3) conda activate TransformerST-deconv  
(4) pip install -r requirements.txt

**2. Instructions: Demo on mouse lung data.**   
 
**2.1 Raw data**

Raw data should be placed in the folder data.

we take the mouse lung data for example, which is in data/Lung/A1. Please use the link to get the mouse lung data
https://drive.google.com/drive/folders/1anVCPRrPIB2jV6DTn6UGJL5X0pErdUqg?usp=share_link


**2.2 Cell type identification at spot resolution with Mouse Lung data**

The TransformerST model is implemented in Mouse_Lung_histology.py. When running TransformerST, the data path should be specified, please modify the --data_root  and --proj_list here. In addition, the parameter --save_root should also be modified to save the experimental results.

The defination of each argument in Mouse_Lung_histology.py is listed below. The parameters are chosen based on the experiments.

'--k', type=int, default=20, help='parameter k in spatial graph'  
'--knn_distanceType', type=str, default='euclidean',help='graph distance type: euclidean/cosine/correlation'  
'--epochs', type=int, default=1000, help='Number of epochs to train.'  
'--cell_feat_dim', type=int, default=3000, help='Dim of input genes'  
'--feat_hidden1', type=int, default=512, help='Dim of DNN hidden 1-layer.'  
'--feat_hidden2', type=int, default=128, help='Dim of DNN hidden 2-layer.'  
'--gcn_hidden1', type=int, default=128, help='Dim of GCN hidden 1-layer.'  
'--gcn_hidden2', type=int, default=64, help='Dim of GCN hidden 2-layer.'  
'--p_drop', type=float, default=0.2, help='Dropout rate.'  
'--using_dec', type=bool, default=True, help='Using DEC loss.'  
'--using_mask', type=bool, default=False, help='Using mask for multi-dataset.'  
'--feat_w', type=float, default=1, help='Weight of DNN loss.'  
'--gcn_w', type=float, default=1, help='Weight of GCN loss.'  
'--dec_kl_w', type=float, default=1, help='Weight of DEC loss.'  
'--gcn_lr', type=float, default=0.001, help='Initial GNN learning rate.'  
'--gcn_decay', type=float, default=0.0001, help='Initial decay rate.'  
'--dec_cluster_n', type=int, default=10, help='DEC cluster number.'  
'--dec_interval', type=int, default=20, help='DEC interval nnumber.'  
'--dec_tol', type=float, default=0.00, help='DEC tol.'  
'--eval_resolution', type=int, default=1, help='Eval cluster number.'  
'--eval_graph_n', type=int, default=20, help='Eval graph kN tol.' 
3. Following the implementation of the graph transformer and vision transformer models, the essential components of our deconvolution mixture of experts model can be found within the 'MOE' directory, serving as a reference for further examination.
