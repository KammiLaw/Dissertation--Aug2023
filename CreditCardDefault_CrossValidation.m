data = readtable('default of credit card clients.csv');
X = data{:,2:end-1};
y = data{:,'Y'};

% Prepare a partition for cross-validation
cv = cvpartition(y, 'KFold', 5);

% Initialize a variable to hold accuracy results
accuracy = zeros(cv.NumTestSets, 1);

% Loop over each fold
for i = 1:cv.NumTestSets
    % Training/test indices for this fold
    trainingIdx = cv.training(i);
    testIdx = cv.test(i);
    
    % Train a decision tree model
    model_2 = fitctree(X(trainingIdx, :), y(trainingIdx));

    % Predict the labels of test set
    predictions = predict(model_2, X(testIdx, :));
    
    % Evaluate the model
    accuracy(i) = sum(predictions == y(testIdx)) / length(y(testIdx));
end

% Calculate mean accuracy across all folds
mean_accuracy = mean(accuracy);
disp(['Average accuracy over all folds: ', num2str(mean_accuracy)]);