% Importing default of credit card clients data 
data = readtable('default of credit card clients.csv');

% Features and labels
X = data{:,2:end-1};
y = data{:,'Y'}; % default payment next month

% Split the data into training and test sets
cv = cvpartition(size(data, 1), 'HoldOut', 0.2);
idx = cv.test;

% Separate to training and test data
trainData = X(~idx,:);
testData = X(idx,:);
trainLabels = y(~idx);
testLabels = y(idx);

% Train a decision tree model
model_1 = fitctree(trainData, trainLabels);

% Predict the labels of test set
predictions = predict(model_1, testData);

% Evaluate the model
accuracy = sum(predictions == testLabels) / length(testLabels);
disp(['Accuracy: ', num2str(accuracy)]);