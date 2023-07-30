% Importing St. Louis Fed Financial Stress Index (STLFSI4) 
STLFSI_data = readtable('STLFSI4.csv');
STLFSI_dates = datetime(STLFSI_data.DATE, 'InputFormat', 'yyyy-MM-dd');  
STLFSI = STLFSI_data.STLFSI4;

% Importing 10-Year Treasury Constant Maturity Minus 3-Month Treasury Constant Maturity (T10Y3M) 
T10Y3M_data = readtable('T10Y3M.csv');
T10Y3M_dates = datetime(T10Y3M_data.DATE, 'InputFormat', 'yyyy-MM-dd'); 
T10Y3M = T10Y3M_data.T10Y3M;
[common_dates, STLFSI_idx, T10Y3M_idx] = intersect(STLFSI_dates, T10Y3M_dates);
STLFSI_aligned = STLFSI(STLFSI_idx);
T10Y3M_aligned = T10Y3M(T10Y3M_idx);

% Linear regression model ( y = β0 + β1.*x + ε)
linearModel = fitlm(T10Y3M_aligned, STLFSI_aligned);
disp(linearModel);
% Predict STLFSI4 using the model
STLFSI_predict = predict(linearModel, T10Y3M_aligned);

% Scatter plot T10Y3M vs. STLFSI4
figure;
scatter(T10Y3M_aligned, STLFSI_aligned, 'b');
hold on;
plot(T10Y3M_aligned, STLFSI_predict, 'r-', 'LineWidth', 2);
xlabel('10-Year Treasury Constant Maturity Minus 3-Month Treasury Constant Maturity');
ylabel('STLFSI4');
legend('Data', 'Linear Regression', 'Location', 'best');
title('Regression of STLFSI4 on Yield Curve');
hold off;