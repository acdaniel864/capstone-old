# Investment-Grade Wine - README 
Capstone Aaran Daniel
Predictive Modelling and Application Development for Investment-Grade (Fine) Wine

## Problem Statement
Working for a wine investment advisory company, they would like predict future prices of investment-grade wines to successfully advise customers on investments. The task involves building a predictive model and a back-of-house application for the sales team to search the portfolio for wines and provide historical returns to potential customers/investors.

## Primary Aim
Develop a predictive model using historical price data of investment-grade wines to forecast future price appreciation.

## Secondary Aims
- Identify key price predictive features of investment-grade wines.<br>
- Discover additional trends within the investment-grade wine portfolio for sales/marketing strategies when onboarding new customers.<br>
- Identify currently undervalued wines and those beneficial for the sales/marketing strategies, emphasizing:<br>
  1. Growth of back vintages to reassure investors.<br>
  2. Data-driven selection of great wines at great valuations to persuade customers to invest.<br>
- Build a back-of-house application for quick portfolio searches and providing investment returns to potential investors.<br>

## Metric for Success
TBD - some research required to define the accuracy and criteria that would validate the model's readiness for production use.
For now I think within 10%+- (via RMSE) of actual price. 

## Process

### 1. Data Collection
- **Source**: Wine Compare API, access confirmed awaiting access key
- **Key Features**: price history, current market price, vintage, region, producer, bottle size, weather data, critics ratings, grape variety, rarity, production quantities, brand power, regional vintage quality, optimum drinking age. Optional additions: Cru Classe status (or equivalent), Robert Parker score, Averaged Other Critics' score, Wine-Searcher rank / Google reach, Weighted production levels, Liquidity as evidenced by Liv-ex bid/offer spread, Supply to market over time.
- **Core Wine Regions to focus on**: Napa Valley (US), Piedmont and Tuscany (Italy), Bordeaux, Burgundy, Champagne, and Rhone (France).

### 2. Data Preparation
- Combine data sources, handle missing values/outliers.
- Feature Engineering: 
        - Climate effects (with a combination of year and region info)
        - price-to-rating ratio
        - historical price volatility: via standard deviation or beta. 
                - Standard deviation: calculate the standard deviation of each wine and split wines into binary volitile or not, or use standard deviation as a feature?
                - Beta: Use the liv ex 100 to set a benchmark, calculate covariance between the wines and the bench mark. beta = covariance / market_variance. (A beta greater than 1 indicates that the stock is more volatile than the market, while a beta less than 1 suggests it is less volatile.)
        - Critic Name and Critic score interaciton variable. e.g. 
- Scale features for unbiased model training.
- Are critics ratings more/less predictive of prices within certain regions that are more under the spotlight?

### 3. EDA - Aggregate Level
- Calculate summary statistics like mean, median, and standard deviation of price changes.
- Explore correlations between price changes and potential features like vintage, region, grape variety, critic ratings, and auction history.
- Identify trends in the data particularly those which can be of use to sales and marketing teams. 
- Find initial coefficients with LR.

### 4. EDA - Individual Wine Level
- Investigate price trends for individual wines over time.
- Analyse the distribution of price changes for different wine characteristics.
- Visualise relationships between features and price changes for individual wines.
- Find initial coefficients with LR.

### 5. Modelling
#### Aggregate Level
    - Linear regression - analyse the linear relationship between features and price appreciation.
    - Decision trees - identify important features and potential non-linear relationships.
    - Random forests - improve prediction accuracy by combining multiple decision trees.
    - Gradient Boosting (XGB) - leverage sequential learning to improve upon individual models.
    - Forecasting SARIMAX, ARIMA, VAR, Prophet, RNNs.
- Consider ensemble methods to combine predictions from multiple models for improved accuracy.

#### Individual Wine Level
    * ARIMA or SARIMAX: Capture trends and seasonality in individual wine prices.
    * Long Short-Term Memory (LSTM) networks: Handle complex temporal dependencies in price data.
    * VAR models
    * RNNs if enough data 
    * Explore ensemble methods: Combine predictions from different models (e.g., aggregate and individual) for improved accuracy.
    * Considering clustering spark AWS 
    * Look into FB prophet depending on amount of data available 
    * XGBoost for time series  https://machinelearningmastery.com/xgboost-for-time-series-forecasting/ 
    * look into SHAP https://shap.readthedocs.io/en/latest/ 
    * Prophet - do we see varience between days. 

### 6. Evaluation and Iteration
- Use metrics like root mean squared error (RMSE) or R-squared to assess prediction accuracy.
- Use cross-validation techniques specific to time series data, like Time Series Split or Walk Forward Validation
- Pros and cons of initial iteration of the model. 
- Implement monitoring to track model performance over time, and plan for periodic retraining as new data becomes available.

## Potential Challenges
- Insufficient data or attributes.
- Lack of domain knowledge for market segmentation.
- Time series modelling difficulties and unpredictability. 
- Possible need for bespoke models per wine category.
- Computational and resource limitations for advanced models.
- Risk of models predicting current trends without offering new insights.

