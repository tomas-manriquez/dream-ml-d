# What are the most used methods?

**answer:** The most commonly used time-series preprocessing methods include:

**Data Cleaning:**
- Missing value imputation (forward fill, backward fill, linear/spline interpolation, seasonal decomposition-based imputation)
- Outlier detection and handling (IQR method, isolation forest, statistical bounds, k-means clustering)
- Noise reduction (moving averages, exponential smoothing, filtering)
- **Extra**: Merge de variables en una sola para formar fechas

**Data Transformation:**
- Differencing (first-order, seasonal differencing) to achieve stationarity
- Scaling and normalization (min-max scaling, z-score standardization, robust scaling)
- Logarithmic transformation for variance stabilization
- Box-Cox transformation for normality

**Feature Engineering:**
- Lag features creation
- Rolling statistics (mean, std, min, max)
- Seasonal decomposition (trend, seasonal, residual components)
- Fourier transforms for frequency domain analysis

**Data Structuring:**
- Time index formatting and sorting
- Resampling (upsampling/downsampling)
- Window-based transformations for supervised learning

**relevant sources:**
* https://www.sciencedirect.com/science/article/pii/S2307187724000452: "Data preprocessing is a time consuming and complex phase that lacks a unified and structured approach. We survey data preprocessing techniques under different categories to provide an extended and structured scope of data preprocessing relevant to numerical time-series data" 
* https://machinelearningmastery.com/machine-learning-data-transforms-for-time-series-forecasting/: "Given a univariate time series dataset, there are four transforms that are popular when using machine learning methods to model and make predictions"  - covering differencing, power transforms, normalization, and standardization
* https://medium.com/enjoy-algorithm/pre-processing-of-time-series-data-c50f8a3e7a98: "Time Series preprocessing techniques have a significant influence on data modeling accuracy"  - discusses structuring, missing values, denoising, and outlier detection
* https://towardsdatascience.com/preprocessing-time-series-data-for-supervised-learning-2e27493f44ae/: "Many ML practitioners are already familiar and experienced with standard algorithms like Decision Trees, Ensemble Tree-based models, Artificial Neural Network Regression"  - covers supervised learning preprocessing approaches

# What are the best practices in this step?

**answer:** The best practices for time-series preprocessing include:

**Sequential Processing Order:**
1. Handle missing values first
2. Remove/handle outliers 
3. Apply transformations for stationarity
4. Scale/normalize data
5. Engineer features last

**Domain-Specific Considerations:**
- Understand temporal dependencies and avoid data leakage from future observations
- Use rolling window statistics rather than global statistics to prevent look-ahead bias
- Preserve temporal order during train/test splits
- Apply same preprocessing steps to training and test data using training-derived parameters

**Stationarity Requirements:**
- Test for stationarity using statistical tests (ADF, KPSS)
- Apply appropriate transformations (differencing, detrending) when needed
- Validate transformations don't over-difference the data

**Handling Missing Values:**
- Choose method based on missingness pattern (MCAR, MAR, MNAR)
- Use interpolation methods that respect temporal structure
- Document and justify imputation choices

**Scaling and Normalization:**
- Use robust methods when outliers are present
- Estimate normalization parameters from training data only
- Consider rolling window normalization for non-stationary variance

**relevant sources:**
* https://www.numberanalytics.com/blog/ultimate-guide-stationarity-time-series: "Non-stationary inputs can lead to spurious regressions. Always pre-test and transform before model fitting" 
* https://machinelearningmastery.com/normalize-standardize-time-series-data-python/: "Any transforms performed to data prior to training must also be performed to test or any other data" 
* https://www.mdpi.com/1999-4893/17/8/332: "Understanding the properties of the data, such as stationarity and distribution, is essential for selecting appropriate preprocessing techniques" 
* https://medium.com/@bhatadithya54764118/day-35-time-series-data-preprocessing-handling-temporal-data-c7e5063752de: "Handling time-series data is an art that blends domain knowledge with statistical techniques" 

# How much do I need to know about the data for this step?

**answer:** Domain knowledge requirements for time-series preprocessing are substantial and operate at multiple levels:

**Essential Domain Understanding:**
- **Data generation process**: How and why the data was collected, sampling frequency, measurement conditions
- **Temporal patterns**: Expected seasonality, trends, cyclical behaviors specific to the domain
- **Missing data mechanisms**: Whether missing values are random, systematic, or domain-specific
- **Outlier context**: What constitutes a true anomaly vs. natural variation in the domain

**Domain-Specific Preprocessing Decisions:**
- **Feature engineering**: What domain-relevant features to create (e.g., business cycles in finance, weather patterns in energy)
- **Transformation selection**: Whether to use log transforms for multiplicative seasonality, or Box-Cox for variance stabilization
- **Imputation strategy**: Forward-fill for slowly changing processes vs. interpolation for continuous phenomena
- **Outlier handling**: Whether to remove anomalies or treat them as important events

**Business Context Requirements:**
- **Forecast horizon and use case**: Short vs. long-term predictions require different preprocessing approaches
- **Model interpretability needs**: Some domains require explainable preprocessing steps
- **Real-time constraints**: Whether preprocessing must be computationally efficient for streaming data
- **Regulatory requirements**: Some industries have specific data handling requirements

**Technical Understanding Needed:**
- **Statistical properties**: Understanding of stationarity, autocorrelation, heteroscedasticity in domain context
- **Data quality issues**: Common problems in domain-specific data collection systems
- **External factors**: Events that affect the time series (holidays, policy changes, market conditions)

**relevant sources:**
* https://www.geeksforgeeks.org/role-of-domain-knowledge-in-data-science/: "Domain Insight: Domain knowledge informs data preprocessing, helping to identify meaningful attributes and streamline the cleaning process for more efficient transformation" 
* https://blog.ml.cmu.edu/2020/08/31/1-domain-knowledge/: "Domain knowledge can help us understand how our data are collected and hence, the appropriate methods for preprocessing" 
* https://www.mdpi.com/1999-4893/17/8/332: "MNAR typically has more null values compared to MCAR and can only be identified through domain knowledge" 
* https://pmc.ncbi.nlm.nih.gov/articles/PMC10457853/: "Feature engineering can be very important in the data preprocessing stage before feeding data to the Deep Learning models by significantly reducing the computational requirements for unnecessary features"