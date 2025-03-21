# TLOB: A Novel Transformer Model with Dual Attention for Stock Price Trend Prediction with Limit Order Book Data
This is the official repository for the paper TLOB: A Novel Transformer Model with Dual Attention for Stock Price Trend Prediction with Limit Order Book Data.
![TLOB Architecture](https://github.com/LeonardoBerti00/TLOB/blob/main/Architecture.png)

## Abstract
Stock Price Trend Prediction (SPTP) based on Limit Order Book
(LOB) data is a fundamental challenge in financial markets. Despite advances in deep learning, existing models fail to generalize across different market conditions and struggle to predict short-term trends reliably. Surprisingly, by adapting a simple MLP-based architecture to LOB, we show that we surpass SoTA performance, thus challenging the necessity of complex architectures. Unlike past work that shows robustness issues, we propose TLOB, a transformer-based model that uses a dual attention mechanism to capture spatial and temporal dependencies in LOB data. This allows it to adaptively focus on the market microstructure, making it particularly effective for longer-horizon predictions and volatile market conditions. We also introduce a new labeling method that improves on previous ones, removing the horizon bias. We evaluate LOB’s effectiveness using the established FI-2010 benchmark, which exceeds the state-of-the-art by an average of 3.7 F1-score(%). Additionally, TLOB shows improvements on Tesla and Intel with a 1.3 and 7.7 increase in F1-score(%), respectively. Additionally, we empirically show how stock price predictability has declined over time (-6.68 absolute points in F1-score(%)), highlighting the growing market efficiencies. Predictability must be considered in relation to transaction costs, so we experimented with defining trends using an average spread, reflecting the primary transaction cost. The resulting performance deterioration  underscores the complexity of translating trend classification into profitable trading strategies. We argue that our work provides new insights into the evolving landscape of stock price trend prediction and sets a strong foundation for future advancements in financial AI.

# Getting Started 
These instructions will get you a copy of the project up and running on your local machine for development and reproducibility purposes.

## Prerequisities
This project requires Python and pip. If you don't have them installed, please do so first. It is possible to do it using conda, but in that case, you are on your own. You can use both windows and linux, but to use linux you need to change...  

## Installing
To set up the environment for this project, follow these steps:

1. Clone the repository:
```sh
git clone https://github.com/LeonardoBerti00/TLOB.git
```
2. Navigate to the project directory
3. Create a virtual environment:
```sh
python -m venv env
```
4. Activate the new pip environment in windows:
```sh
env\Scripts\activate
```
or in linux:
```sh
source env/Scripts/activate
```
5. Download the necessary packages:
```sh
pip install -r requirements.txt
```

# Reproduce the results
To reproduce the results follow the following steps:

1. In data/checkpoints/TLOB/HuggingFace/ you can find the four checkpoints for FI-2010, the checkpoints for TSLA and INTC did not fit in the free GitHub repository size. If you need also the other checkpoints you can contact me. 
2. Inside the config file, you need to set Dataset to FI-2010, set the horizons to 1, 2, 5, or 10 finally set the experiment type to EVALUATION. Everything should already be set.
3. Now run:
```sh
python main.py +model=tlob hydra.job.chdir=False
```
After doing the first run, both for training or reproducing, you can set config.Experiment.is_data_preprocessed to True.
Note that the horizons in the paper are an order of magnitude higher because in the paper the value represent the horizons before the sampling process of the dataset. In fact, the dataset is sampled every 10 events. 

# Training
If your objective is to train a TLOB or MLPLOB model or implement your model you should follow those steps.

## Data 
If you have some LOBSTER data you can follow those steps:
1. The format of the data should be the same of LOBSTER: f"{year}-{month}-{day}_34200000_57600000_{type}" and the data should be saved in f"data/{stock_name}/{stock_name}_{year}-{start_month}-{start_day}_{year}-{end_month}-{end_day}". Type can be or message or orderbook.
2. Inside the config file, you need to set the name of the training stock and the testing stocks, and also the dataset to LOBSTER. Currently you can add only one for the training but several for testing. 
3. You need to start the pre-processing step, to do so set config.is_data_preprocessed to False and run python main.py

Otherwise, if you want to train and test the model with the Benchmark dataset FI-2010 you need, inside the config file, set Dataset to FI-2010 and set the horizons to 1, 2, 5, or 10, and the experiment type to TRAINING. Note that the horizons in the paper are an order of magnitude higher because in the paper the value represent the horizons before the sampling process of the dataset. In fact, the dataset is sampled every 10 events. After doing the first run, both for training or reproducing, you can set config.Experiment.is_data_preprocessed to True.


## Training a TLOB, MLPLOB, DeepLOB or BiNCTABL Model 
To train a TLOB, MLPLOB, DeepLOB or BiNCTABL Model, you need to set the type variable in the config file to TRAINING, then run this command:
```sh
python main.py +model={model_name} hydra.job.chdir=False
```
you can see all the model names in the config file. 

## Implementing and Training a new model 
To implement a new model, follow these steps:
1. Implement your model class in the models/ directory. Your model class will take in input an input of dimension [batch_size, seq_len, num_features], and should output a tensor of dimension [batch_size, 3].
2. add your model to pick_model in utils_models.
3. Update the config file to include your model and its hyperparameters. If you are using the FI-2010 dataset, It is suggested to set the hidden dim to 40 and the hp all_features to false if you want to use only the LOB as input or if you want to use the LOB and market features the hidden dim should be 144 and all features true. If you are using LOBSTER data, it is suggested to set the hidden dim to 46 and all features to true to use LOB and orders, while if you want to use only the LOB set all features to False. 
4. Add your model with cs.store, similar to the other models
5. Run the training script:
```sh
python main.py +model={your_model_name} hydra.job.chdir=False
```
6. You can set whatever configuration using the hydra style of prompt.
7. A checkpoint will be saved in data/checkpoints/ 

Optionally you can also log the run with wandb or run a sweep, changing the config experiment options.

# Results
MLPLOB and TLOB outperform all the other SoTA deep learning models for Stock Price Trend Prediction with LOB data for both datasets, FI-2010 benchmark and TSLA-INTC.
![FI-2010 results](https://github.com/LeonardoBerti00/TLOB/blob/main/fI-2010.png)
![TSLA and INTC results](https://github.com/LeonardoBerti00/TLOB/blob/main/tslaintc.png)

# Citation
```sh
@misc{berti2025tlobnoveltransformermodel,
      title={TLOB: A Novel Transformer Model with Dual Attention for Stock Price Trend Prediction with Limit Order Book Data}, 
      author={Leonardo Berti and Gjergji Kasneci},
      year={2025},
      eprint={2502.15757},
      archivePrefix={arXiv},
      primaryClass={q-fin.ST},
      url={https://arxiv.org/abs/2502.15757}, 
}
```
