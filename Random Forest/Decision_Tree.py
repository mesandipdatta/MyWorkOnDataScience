"""
We will train a decision tree
classifier using the Internet Advertisements Data Set from http://archive.ics.uci.
edu/ml/datasets/Internet+Advertisements, which contains data for 3,279 images.
The proportions of the classes are skewed; 459 of the images are advertisements and
2,820 are content. Decision tree learning algorithms can produce biased trees from data
with unbalanced class proportions; we will evaluate a model on the unaltered data set
before deciding if it is worth balancing the training data by over- or under-sampling
instances. The explanatory variables are the dimensions of the image, words from the
containing page's URL, words from the image's URL, the image's alt text, the image's
anchor text, and a window of words surrounding the image tag. The response variable
is the image's class. The explanatory variables have already been transformed into
feature representations. The first three features are real numbers that encode the width,
height, and aspect ratio of the images. The remaining features encode binary term
frequencies for the text variables. In the following sample, we will grid search for the
hyperparameter values that produce the decision tree with the greatest accuracy,
and then evaluate the tree's performance on a test set:
"""
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV

"""
First we read the .csv file using pandas. The .csv does not have a header row,
so we split the last column containing the response variable's values from the
features using its index:
"""

if __name__ == '__main__':
	df = pd.read_csv('data/ad.data', header=None)

	explanatory_variable_columns = set(df.columns.values)

	response_variable_column = df[len(df.columns.values)-1]

	# The last column describes the targets
	explanatory_variable_columns.remove(len(df.columns.values)-1)

	y = [1 if e == 'ad.' else 0 for e in response_variable_column]

	X = df[list(explanatory_variable_columns)]


"""
	We encoded the advertisements as the positive class and the content as the negative
class. More than one quarter of the instances are missing at least one of the values
for the image's dimensions. These missing values are marked by whitespace and a
question mark. We replaced the missing values with negative one, but we could have
imputed the missing values; for instance, we could have replaced the missing height
values with the average height value:
"""

X.replace(to_replace=' *\?', value=-1, regex=True, inplace=True)

X_train, X_test, y_train, y_test = train_test_split(X, y)

"""
We created a pipeline and an instance of DecisionTreeClassifier. Then, we set
the criterion keyword argument to entropy to build the tree using the information
gain heuristic:
"""

pipeline = Pipeline([
					('clf', 
					  DecisionTreeClassifier(criterion='entropy'))
					])


"""
Next, we specified the hyperparameter space for the grid search:
"""

parameters = {'clf__max_depth': (150, 155, 160),
			  'clf__min_samples_split': (1, 2, 3),
			  'clf__min_samples_leaf': (1, 2, 3)
  			 }


"""
We set GridSearchCV() to maximize the model's F1 score:
"""
grid_search = GridSearchCV(pipeline, 
						   parameters, 
						   n_jobs=-1, 
						   verbose=1, 
						   scoring='f1')

grid_search.fit(X_train, y_train)

print 'Best score: %0.3f' % grid_search.best_score_

"""Best score: 0.878"""

print 'Best parameters set:'

best_parameters = grid_search.best_estimator_.get_params()

for param_name in sorted(parameters.keys()):
		print '\t%s: %r' % (param_name, best_parameters[param_name])

predictions = grid_search.predict(X_test)
print classification_report(y_test, predictions)


"""
Fitting 3 folds for each of 27 candidates, totalling 81 fits
[Parallel(n_jobs=-1)]: Done 1 jobs | elapsed: 1.7s
[Parallel(n_jobs=-1)]: Done 50 jobs | elapsed: 15.0s
[Parallel(n_jobs=-1)]: Done 71 out of 81 | elapsed: 20.7s
remaining: 2.9s
[Parallel(n_jobs=-1)]: Done 81 out of 81 | elapsed: 23.3s finished


Best score: 0.878

Best parameters set:
clf__max_depth: 155
clf__min_samples_leaf: 2
clf__min_samples_split: 1


	precision recall f1-score support
0 	0.97 	  0.99 	 0.98     710
1 	0.92 	  0.81 	 0.86     110

avg / total 0.96 0.96 0.96 820


The classifier detected more than 80 percent of the ads in the test set, and
approximately 92 percent of the images that it predicted were ads were truly ads.
Overall, the performance is promising; in following sections, we will try to modify
our model to improve its performance.


"""