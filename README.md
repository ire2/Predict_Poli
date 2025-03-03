# API for Predicting Attitudes Towards Marijuana 


## Model Details

- **Developed by**: Ignacio Estrada Cavero  
- **Description**: This is a Random Forest model designed to predict whether individuals believe marijuana should be legalized. It is based on demographic, behavioral, and attitudinal data collected in the [2022 General Social Survey (GSS)](https://gss.norc.org/us/en/gss/get-documentation.html). The model utilizes 10 key features derived from survey responses to make predictions.  
- **Feature Engineering/Data Preprocessing**: The feature engineering process included systematic handling of missing data, SMOTE for addressing class imbalance, normalization of numerical features, and one-hot encoding of categorical variables. For more detailed information, refer to the "Feature Engineering" section.  
- **Version**: This model is identified as `r v$metadata$version` and was published on `r v_meta$created`.  
- **Citation/License**: This model adheres to the usage terms outlined by the General Social Survey’s data-sharing policy. Contact Ignacio Estrada Cavero at <ire2@cornell.edu> for citation or licensing details.  
- **Questions**: For any inquiries regarding this model, please contact Ignacio at <ire2@cornell.edu>.  

## Intended Use

- **Primary Uses**:  
  - Sociological research to explore attitudes toward marijuana legalization.  
  - Academic purposes for learning and demonstrating Random Forest modeling techniques.  
  - Prototyping workflows for predicting sociological outcomes using machine learning.  

- **Primary Users**:  
  - Students and educators studying predictive modeling and sociological datasets.  
  - Researchers aiming to understand the demographic and attitudinal factors influencing public opinions.  

- **Out of Scope**:  
  - High-stakes decision-making, such as influencing public policy or legislative actions.  
  - Commercial use in marketing or product development without further refinement and validation.  

---

## Important Aspects/Factors

- **Aspects or Factors Relevant to the Context of This Model**:  
  - **Demographic**: Factors like age, gender, political affiliation, and employment status are crucial in shaping attitudes toward marijuana legalization.  
  - **Behavioral**: Patterns in work status, political ideology, and previous voting behavior influence predictions.  
  - **Technical**: Class imbalance in the dataset and missing data handling significantly impact the model's accuracy and reliability.  

- **Evaluated Aspects**:  
  - **Class Imbalance**: Addressed through SMOTE to balance the two predicted classes (`should be legal` and `should not be legal`).  
  - **Feature Importance**: Identified the most influential predictors, such as political views and work status.  
  - **Performance Metrics**: Evaluated accuracy, Cohen’s kappa, and confusion matrix to assess model effectiveness.  

---

## Metrics

- **Metrics Used to Evaluate the Model**:  
  - **Accuracy**: Measures the proportion of correct predictions out of all predictions made.  
  - **Cohen’s Kappa**: Accounts for agreement beyond chance, offering insight into performance on imbalanced datasets.  
  - **Confusion Matrix**: Summarizes prediction performance for each class to highlight areas of strength and weakness.  

- **Computation**:  
  - Metrics were calculated using a validation dataset with 5 Fold CV sampling to ensure representativeness.  
  - The `yardstick` package in R was used to compute accuracy, Cohen’s kappa, ROC AUC and confusion matrix values.  

- **Rationale for Metric Choice**:  
  - **Accuracy** provides a straightforward measure of overall correctness.  
  - **ROC AUC** is useful for evaluating the model's ability to discriminate between classes.
  - **Cohen’s Kappa** offers a more nuanced perspective, especially in imbalanced datasets.  
  - The **confusion matrix** provides detailed insights into errors and successes across classes, guiding further improvements.  

---

## Training Data & Evaluation Data

- **Training Dataset**:  
  - The model was trained on a subset of the 2022 General Social Survey (GSS) dataset.  
  - Key preprocessing steps included:
    - Imputation of missing values using median (numeric) and mode (categorical).  
    - One-hot encoding of categorical features for compatibility with the Random Forest model.  
    - Normalization of numeric features.  
    - Balancing the target variable (`grass`) using SMOTE to address class imbalance.  

- **Evaluation Dataset**:  
  - A stratified subset of the GSS dataset was reserved for validation.  
  - This validation set followed the same preprocessing steps as the training data to ensure consistency in evaluation.  

- **Prototype**:  
  - The dataset’s prototype includes the following key features used in the model:  
```{r}
    glimpse(v$prototype)
```

  - The evaluation dataset used in this model card is a stratified subset of the 2022 General Social Survey (GSS). This subset was selected to maintain representativeness of the population and preserve the proportional distribution of the target variable (`grass`) across its two classes: `should be legal` and `should not be legal`.

- We chose this evaluation data because it ensures:
  - **Representativeness**: The stratified sampling maintains the same distribution of key features and the target variable as the original dataset, making evaluation results reflective of real-world scenarios.  
  - **Consistency**: By using the same preprocessing pipeline as the training dataset, the evaluation dataset eliminates variability caused by differences in data preparation.  
  - **Fair Assessment**: The stratified split avoids bias that could arise from imbalanced or non-representative subsets, providing a reliable estimate of the model’s generalization performance.  


