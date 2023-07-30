data = readtable('default of credit card clients.csv');
X = data{:,2:end-1};
y = data{:,'Y'};
cv = cvpartition(y, 'KFold', 5);

% Initialize a variable to hold AUC results
auc = zeros(cv.NumTestSets, 1);

figure

for i = 1:cv.NumTestSets
    % Training/test indices for this fold
    trainingIdx = cv.training(i);
    testIdx = cv.test(i);
 
    model_4 = fitctree(X(trainingIdx, :), y(trainingIdx));

    % Predict the probabilities of test set
    [~, scores] = predict(model_4, X(testIdx, :));
    
    % Compute the ROC curve
    [Xroc,Yroc,T,AUC] = perfcurve(y(testIdx),scores(:,2),'1');
    
    auc(i) = AUC;
   
    % Plot the ROC curve for this fold
    plot(Xroc, Yroc)
    hold on
end

% Mean AUC across all folds
mean_auc = mean(auc);
disp(['Average AUC over all folds: ', num2str(mean_auc)]);
xlabel('False positive rate') 
ylabel('True positive rate')
title('ROC for Classification by Decision Tree')
legend('Fold 1','Fold 2','Fold 3','Fold 4','Fold 5','Location','Best')