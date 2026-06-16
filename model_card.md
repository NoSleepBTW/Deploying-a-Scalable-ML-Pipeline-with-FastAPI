# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

This model is a binary classifier that predicts whether an individual's annual
income exceeds $50,000 based on U.S. Census demographic data. It is a
scikit-learn `RandomForestClassifier` trained with 100 estimators and a fixed
`random_state` of 42 for reproducibility. Categorical features are encoded with
a one-hot `OneHotEncoder` (configured with `handle_unknown="ignore"`) and the
target is binarized with a `LabelBinarizer`. The trained model, the encoder, and
the label binarizer are serialized with pickle to the `model/` directory. The
model was developed as part of the Udacity "Deploying a Scalable ML Pipeline
with FastAPI" project.

## Intended Use

The model is intended for educational and demonstration purposes, showing an
end-to-end machine learning pipeline that includes training, evaluation on data
slices, and deployment behind a FastAPI REST endpoint. The primary users are
students and engineers learning ML DevOps practices. It is **not** intended for
real-world decisions about individuals, such as lending, hiring, or credit
determinations.

## Training Data

The training data comes from the UCI Census Income ("Adult") dataset, provided
as `data/census.csv`, which contains 32,561 records with 14 demographic
features (e.g. age, workclass, education, marital status, occupation,
relationship, race, sex, hours-per-week, and native-country) plus the `salary`
target label. The data was cleaned to remove stray whitespace. The dataset was
split 80/20 into training and test sets using a stratified split on the
`salary` label with `random_state=42`, so the training set contains roughly
26,048 records.

## Evaluation Data

The model was evaluated on the held-out 20% test split (roughly 6,513 records),
which was processed with the same one-hot encoder and label binarizer fit on the
training data (using `training=False`). In addition to overall metrics,
performance was computed on slices of each categorical feature, with the
per-slice results written to `slice_output.txt`.

## Metrics

The model is evaluated using precision, recall, and F1 (F-beta with beta=1).
On the held-out test set, the model achieved a precision of 0.7327, a recall of
0.6397, and an F1 score of 0.6830.

Performance varies across data slices. For example, on the `sex` feature the
model achieved an F1 score of 0.6847 for Female (precision 0.7638, recall
0.6204) and an F1 score of 0.6827 for Male (precision 0.7274, recall 0.6432).
On the `education` feature, performance ranged from an F1 score of 0.8889 for
individuals with a Doctorate (precision 0.8889, recall 0.8889) down to an F1
score of 0.4904 for individuals with an HS-grad education (precision 0.5975,
recall 0.4159). The complete per-slice metrics for every categorical feature are
available in `slice_output.txt`.

## Ethical Considerations

The dataset encodes sensitive attributes such as race, sex, and native country,
and the underlying 1994 census data reflects historical social and economic
biases. Because the model learns from these patterns, its predictions can
reproduce or amplify those biases, and the slice metrics show that accuracy is
not uniform across demographic groups. The model should never be used to make or
inform decisions that affect real individuals, and any deployment would require a
careful fairness audit and bias mitigation.

## Caveats and Recommendations

The data is from the 1994 U.S. Census and is now outdated; it does not reflect
current income distributions or social conditions. The dataset is also class
imbalanced, with most individuals earning at or below $50K, which depresses
recall on the positive (>50K) class. The model was trained with default
hyperparameters and no tuning, so performance could likely be improved with
hyperparameter search, cross-validation, scaling of continuous features, or
addressing class imbalance. Users should treat this model strictly as a learning
example and validate any changes against the slice metrics in `slice_output.txt`.
