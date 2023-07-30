% Importing Tesla's Adj close price from Jan2019-Dec2022
filename = 'TSLA.csv'; 
dataTable = readtable(filename);
adjClose = dataTable.AdjClose; 
dates = datetime(dataTable.Date,'InputFormat','yyyy-MM-dd');

% Set the short and long window lengths
shortWindow = 20;
longWindow = 100;

% Compute the short and long moving averages
shortMA = movmean(adjClose, shortWindow);
longMA = movmean(adjClose, longWindow);

% Create a trading signal when the short MA is above the long MA
signal = shortMA > longMA;

% Preallocate the positions vector
positions = zeros(size(signal));

% Generate trading orders: buy (1) when signal turns on, sell (-1) when signal turns off
positions([false; diff(signal) == 1]) = 1;
positions([false; diff(signal) == -1]) = -1;

figure
hold on
plot(dates, adjClose)
plot(dates, shortMA)
plot(dates, longMA)
plot(dates(positions == 1), adjClose(positions == 1), 'g^') % plot buy signals
plot(dates(positions == -1), adjClose(positions == -1), 'rv') % plot sell signals
hold off
legend('Adj Close', 'Short-term MA', 'Long-term MA', 'Buy', 'Sell')
title('TESLA Trading Strategy')
xlabel('Date')
ylabel('Price')