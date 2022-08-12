# Model Card


## Model Details

Diego Iglesias created this model. It is a logistic regression using the default hyperparameters in scikit-learn.

## Intended Use

This model should be used to predict the if the salary of a person is above or below 50k, based on socio-demographic attributes.

## Training Data

The data was obtained from the UCI Machine Learning Repository. It consists of 31.561 data points, with 15 attributes, one of these being the label to predict. The data has been split into train and test by a proportion of 80-20. Categorical attributes have been encoded using a One Hot encoder. Continuous attrubutes have not been scaled.

## Evaluation Data

The evaluation data consists of a 20 percent of the original dataset. It has been processed following the sames steps as the training data.

## Metrics

The model was evaluated using the precission, recall, and Fbeta. The results are 0.70, 0.26 and 0.38.

## Ethical Considerations

The model was evaluated of the different slices of sex, race and native country of the person. The performance is similar in terms of sex. In relation to race, we see some underperformance on some levels. The same happens with the native country. This means that this model should be used with caution in order to not incur in a ethical bias.

## Caveats and Recommendations

New iterations of the model should be done to overcome the ethical considerations. New algorthims can be tried.

