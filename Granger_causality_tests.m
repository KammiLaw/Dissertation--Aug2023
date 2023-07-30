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

% Check for stationarity using Augmented Dickey-Fuller test 
[h_STLFSI, pValue_STLFSI] = adftest(STLFSI_aligned);
[h_T10Y3M, pValue_T10Y3M] = adftest(T10Y3M_aligned);

% Difference the series if non-stationary
if h_STLFSI == 0
    STLFSI_aligned = diff(STLFSI_aligned);
end
if h_T10Y3M == 0
    T10Y3M_aligned = diff(T10Y3M_aligned);
end

% Make sure both series have the same length
if length(STLFSI_aligned) > length(T10Y3M_aligned)
    STLFSI_aligned = STLFSI_aligned(2:end);
elseif length(T10Y3M_aligned) > length(STLFSI_aligned)
    T10Y3M_aligned = T10Y3M_aligned(2:end);
end

% Perform the Granger causality test
maxLags = 4;  % Set the maximum number of lags

for numLags = 1:maxLags
    % Regression with both series
    X = [lagmatrix(STLFSI_aligned, 1:numLags) lagmatrix(T10Y3M_aligned, 1:numLags)];
    X = X(numLags+1:end, :);  % Remove rows with NaNs
    Y = STLFSI_aligned(numLags+1:end);
    mdl_full = fitlm(X, Y);

    % Regression with STLFSI alone
    X_STLFSI = lagmatrix(STLFSI_aligned, 1:numLags);
    X_STLFSI = X_STLFSI(numLags+1:end, :);  
    mdl_STLFSI = fitlm(X_STLFSI, Y);

    % Calculate F-statistic
    SSR_full = mdl_full.SSE;
    SSR_STLFSI = mdl_STLFSI.SSE;
    F = ((SSR_STLFSI - SSR_full) / numLags) / (SSR_full / (length(Y) - 2*numLags));

    % Calculate p-value
    p = 1 - fcdf(F, numLags, length(Y) - 2*numLags);

    fprintf('Number of lags: %d, F-statistic: %.4f, p-value: %.4f\n', numLags, F, p);
end


% Scatter Plot T10Y3M vs. STLFSI4
figure;
scatter(T10Y3M_aligned, STLFSI_aligned, 'o');
xlabel('10-Year Treasury Constant Maturity Minus 3-Month Treasury Constant Maturity');
ylabel('STLFSI4');
title('Scatter plot of STLFSI4 vs T10Y3M');
