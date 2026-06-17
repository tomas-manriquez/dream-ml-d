You are an expert developer who specializes in python for machine learning.\
  \
  Id like to add Random Search hyperparameter tuning to my `train_xgboost_model` function in 
  @DREAM-ML-backend/GEML/apiTimeSeries/train.py as a possible input option. Please help me implement the 
  necessary code changes in order to include this behavior.\
  \
  Please think step-by-step in <thinking> tags before you answer. First, analyze the `train_xgboost_model` 
  function in terms of I/O (consider that the function is called by `train_model_logic` in 
  @DREAM-ML-backend/GEML/apiTimeSeries/services.py , and `train_model_logic` itself is called by 
  `train_model` in @DREAM-ML-backend/GEML/apiTimeSeries/views.py ). Then, think about what changes are 
  necessary in order to implement Random Search. Then, think about the best way to implement this in the 
  code. If you need any further context or clarification, please ask before you continue. You may keep an 
  internal to-do list

---

  You are an expert developer who specializes in python, React and javascript.\
  \
  Ive just implemented Random Search for both of my time series forecasting models in the backend. What
  changes are necessary in the frontend, particularly in the user form in the component at
  @DREAM-ML-frontend/frontend/src/components/TSTrainCard.jsx so that the user can define its own query?
  Please help me implement the necessary code changes for this.\
  \
  Please think step-by-step in <thinking> tags before you answer. First, analyze the backend implementation
  in terms of I/O and how they use the request parameters (the data flow is: `train_model` in
  @DREAM-ML-backend/GEML/apiTimeSeries/views.py , then `train_model_logic` in
  @DREAM-ML-backend/GEML/apiTimeSeries/services.py , then either `train_arima_model` or
  `train_xgboost_model` in @DREAM-ML-backend/GEML/apiTimeSeries/train.py ). Then, analyze the React
  component in @DREAM-ML-frontend/frontend/src/components/TSTrainCard.jsx . Then, think about what changes
  are necessary in order to include Random Search as an user option. Then, think about the best way to
  implement this changes in the code. If you need any further context or clarification, please ask before
  you continue. You may keep an internal to-do list.