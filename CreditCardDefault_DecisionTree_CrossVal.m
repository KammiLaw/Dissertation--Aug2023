% IMport data 
data = readtable('default of credit card clients.csv');
X = data{:,2:end-1};
y = data{:,'Y'};

cv = cvpartition(y, 'KFold', 5);
accuracy = zeros(cv.NumTestSets, 1);

% Define hyperparameters to optimize
hyperparameterOptimizationOptions = struct('AcquisitionFunctionName', 'expected-improvement-plus', 'MaxObjectiveEvaluations', 30);
params = hyperparameters('fitctree', X, y);

% Loop over each fold
for i = 1:cv.NumTestSets
    % Training/test indices for this fold
    trainingIdx = cv.training(i);
    testIdx = cv.test(i);
    
    % Train a decision tree model with hyperparameter optimization
    model_3 = fitctree(X(trainingIdx, :), y(trainingIdx), 'OptimizeHyperparameters', params, 'HyperparameterOptimizationOptions', hyperparameterOptimizationOptions);

    % Predict the labels of test set
    predictions = predict(model_3, X(testIdx, :));

    accuracy(i) = sum(predictions == y(testIdx)) / length(y(testIdx));
end

% Mean accuracy across all folds
mean_accuracy = mean(accuracy);
disp(['Average accuracy over all folds: ', num2str(mean_accuracy)]);