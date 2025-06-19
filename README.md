# UrbanFloodCast

## Urban flood forecasting benchmark dataset
![UrbanFloodCastBench](https://github.com/HydroPML/UrbanFloodCast/blob/main/Figures/Figures02.png)
Urban flood forecasting benchmark dataset is introduced for evaluating various neural urban flood modeling and simulation methods. Two urban areas in Berlin, Germany are considered. For both areas, simulations are conducted using 125 distinct design storm rainfall events. The dataset is publicly available on Zenodo (https://zenodo.org/records/14207323).
## Deep Neural Operator (DNO)
![DNO](https://github.com/HydroPML/UrbanFloodCast/blob/main/Figures/Figures11.png)
A deep neural operator (DNO) is designed for fast, accurate, and resolution-invariant urban flood forecasting. The DNO features an enhanced Fourier layer with skip connections for improved memory efficiency, alongside a deep encoder-decoder framework and an urban-embedded residual loss to enhance modeling effectiveness. 
### Usage
To use the DNO, execute the following steps:
   
1. **Download traning, validation, and test data. Put in the corresponding folder.**
   
2. **Please use DNO/TIFF2PT to convert TIFF files into PT files, thereby streamlining model training and data loading processes.**

3. **Please change the corresponding path in DNO/DNO_main.py to your path.**

4. **Run the DNO/DNO_main.py to train and test your model.**
