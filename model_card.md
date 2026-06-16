# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
I used a Random Forest Classifier from scikit-learn for this project. It is a binary classifier that predicts whether a person makes over $50K a year based on census data. The model was trained with 100 estimators and a random state of 42. For preprocessing, I used a `OneHotEncoder` (with `handle_unknown="ignore"`) for the categorical features and a `LabelBinarizer` for the target variable. The trained model and the encoders are saved in the `model/` folder as `.pkl` files. 

## Intended Use
This model was built to fulfill the requirements for the Udacity "Deploying a Scalable ML Pipeline with FastAPI" project. It is strictly for educational purposes to demonstrate how to build an ML pipeline, evaluate data slices, and set up a FastAPI app. It should not be used to make actual real-world decisions regarding lending, hiring, or salaries.

## Training Data
The data is the public UCI Census Income ("Adult") dataset, saved in `data/census.csv`. It has 32,561 rows and 14 demographic features, including age, education, marital status, sex, and hours worked per week. I removed the extra whitespace from the columns and split the data, using 80% of it (about 26,048 rows) for training. The split was stratified based on the target `salary` column to maintain the class distribution.

## Evaluation Data
The remaining 20% of the dataset (about 6,513 rows) was held out and used for testing. I evaluated the model on this test set and also calculated performance metrics across different categorical data slices to see how the model performs on specific sub-groups. The results for these slices are saved in the `slice_output.txt` file.

## Metrics
I evaluated the overall model using precision, recall, and F1 score. On the test set, the results were:
* **Precision:** 0.7327
* **Recall:** 0.6397
* **F1 Score:** 0.6830

Looking at the data slices in `slice_output.txt`, performance changes depending on the specific group. For instance, the F1 score for Female is 0.6847, while for Male it is 0.6827. Education level causes even wider variations; people with a Doctorate had an F1 score of 0.8889, while high school graduates dropped to an F1 score of 0.4904.

## Ethical Considerations
The dataset includes sensitive demographic information like race, sex, and native country. Because this is real historical census data from 1994, it reflects the social and economic inequalities of that time period. The model naturally picks up on these biases, meaning its predictions will not be completely fair across all demographic groups (which is reflected in the slice metrics). It should not be used on real people without serious bias mitigation.

## Caveats and Recommendations
The biggest caveat is that the data is very outdated (1994) and does not represent today's economy. The dataset is also highly class-imbalanced because far more people in the data make under $50K than over it, which negatively impacts the recall for the >50K class. I trained the Random Forest using default hyperparameters, so running a hyperparameter search or utilizing techniques to handle the class imbalance would likely improve the model's performance.