%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% EC7210 Macroeconomics
%
% Code to solve and simulate core RBC model as presented
% in lecture notes 4
%
% Martin Flodén, Fall 2026
% Stockholm University
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all
warning off

global alpha beta gamma delta k_ss h_ss c_ss y_ss logz k1

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Calibrate the model
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
alpha = 0.4;                                        % capital share
delta = 0.02;                                       % depreciation rate of capital
gamma = 2;                                          % parameter in the utility function
ky_target = 12;                                     % target for capital-output ratio
beta = ky_target / (alpha + (1-delta)*ky_target);   % discount factor
rho = 0.95;                                         % persistence of TFP shocks
sigma = 0.009;                                      % std deviation of TFP shocks

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Calculate steady-state values
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
ky_ss = alpha*beta / (1 - beta*(1-delta));          % capital-output ratio
kh_ss = ky_ss^(1/(1-alpha));                        % k/h

h_ss = ((1-alpha)/(1-delta*ky_ss))^(1/(1+gamma));   % hours worked
k_ss = kh_ss*h_ss;                                  % capital stock
y_ss = k_ss/ky_ss;                                  % output
c_ss = y_ss - delta*k_ss;                           % consumption
i_ss = y_ss - c_ss;                                 % investment


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Calculate impulse-response to
% a TFP shock
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
T = 300;                                            % assume that the economy is back in steady state T periods after shock
logz = rho.^(0:T)'*sigma/100;                       % suppose TFP increases by one percent of a std deviation in t=1
k1 = k_ss;                                          % suppose capital stock = k_ss in t=1

% By letting the solver work with the logarithm of 
% k,c and h, we ensure that k, c, and h are always
% positive
xguess = log([k_ss*ones(T,1); ...
              c_ss*ones(T,1); ...
              h_ss*ones(T,1)]);                     % guess the solution for log of k(2):k(T+1), c(1):c(T), h(1):h(T)
                        
opt = optimset('Display','iter','TolX',1e-12);
[x,f] = fsolve(@evaluate_equilibrium_conditions,xguess,opt); % find the solution in response to the TFP shock!

[k,c,h,y,i] = extract_from_x(x);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Calculate policy rules
% Run OLS regression to estimate b0, b1, b2 from:
%   log(k_{t+1}) = b0 + b1*[log(k_t)-log(k_ss)] + b2*log(z_t)
%   log(h_t) = bh0 + bh1*[log(k_t)-log(k_ss)] + b2h*log(z_t)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Y = log(k(2:end));
X = [ones(T,1) (log(k(1:end-1))-log(k_ss)) logz(1:T)];
b = inv(X'*X)*X'*Y;                                     % the vector b is now b0, b1, b2

Y = log(h(1:end-1));
bh = inv(X'*X)*X'*Y;                                  

Y = log(c(1:end-1));
bc = inv(X'*X)*X'*Y;                                  


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Simulate the economy
% Start in steady state
% burnT = periods to simulate before actual simulation
% simT = periods in actual simulation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
burnT = 1000;
simT = 100000;
%rng(7210, 'twister');                                  % Uncomment here to get reproducible simulated moments
simk = zeros(1+burnT+simT,1);                           % k(1), k(2), ...
simz = zeros(1+burnT+simT,1);                           % z(0), z(1), ...
simeps = sigma*randn(burnT+simT,1);                     % eps(1), eps(2), ...
simk(1) = log(k_ss);
simz(1) = 0;

for t = 1:burnT + simT
    simz(t+1) = rho*simz(t) + simeps(t);
    simk(t+1) = [1 simk(t)-simk(1) simz(t+1)]*b;
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Report statistics from simulation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
simK   = exp(simk(burnT+1:end));                        % k(1), ..., k(simT+1)
simTFP = exp(simz(burnT+2:end));                        % z(1), ...
simH = exp([ones(simT,1) log(simK(1:simT))-log(k_ss) log(simTFP)]*bh);
simY = simTFP .* simK(1:simT).^alpha .* simH.^(1-alpha);
simC = simY + (1-delta)*simK(1:simT) - simK(2:simT+1); 
simI = simY - simC;

% Extract business-cycle components with the quarterly HP filter.
logSeries = 100*log([simY simC simI simH]);
lambda = 1600;
[trend,cycle] = hpfilter(logSeries,lambda);

relativeStd = std(cycle) / std(cycle(:,1));
correlations = corrcoef(cycle);
seriesNames = {'Output', 'Consumption', 'Investment', 'Hours'};
fprintf('\nHP-filtered model moments (lambda = 1600)\n');
fprintf('%-12s %24s %24s\n', 'Variable', 'Volatility rel. to output', 'Correlation with output');
for j = 1:numel(seriesNames)
    fprintf('%-12s %24.2f %24.2f\n', seriesNames{j}, relativeStd(j), correlations(j,1));
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Report short sample from the simulation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure
TT = 1000:1050;                                        % short sample
subplot(2,3,1); plot(simY(TT)); title("Output")
subplot(2,3,2); plot(simC(TT)); title("Consumption")
subplot(2,3,3); plot(simH(TT)); title("Hours")
subplot(2,3,4); plot(simTFP(TT)); title("TFP")
subplot(2,3,5); plot(simI(TT)); title("Investment")
subplot(2,3,6); plot(simK(TT)); title("Capital")



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Functions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function f = evaluate_equilibrium_conditions(x)
% x(1:T) is guess for log k(2),..., log k(T+1)
% x(T+1:2*T) is guess for log c(1),..., log c(T)
% x(2*T+1:3*T) is guess for log h(1),..., log h(T)

    global alpha beta gamma delta k_ss h_ss c_ss logz k1
    T = length(x)/3;
    
    [k,c,h,y] = extract_from_x(x);
    r = alpha * y./k - delta;                           % implied interest rate
    w = (1-alpha) * y./h;                               % implied wage rate
    
    evalK = k(2:end) - (1-delta)*k(1:end-1) ...
            - (y(1:T)-c(1:T));                          % dynamics of the capital stock
    evalC = (c(2:end)./c(1:end-1)) - beta*(1+r(2:end)); % Euler equation
    evalH = w(1:T)./c(1:T) - h(1:T).^gamma;                            % optimal hours
    
    f = [evalK; evalC; evalH];

end


function [k,c,h,y,i] = extract_from_x(x)
% x(1:T) is log k(2),..., log k(T+1)
% x(T+1:2*T) is log c(1),..., log c(T)
% x(2*T+1:3*T) is log h(1),..., log h(T)
% Return k(1),...,k(T+1), c(1),...,c(T+1), h(1),...,h(T+1)

    global alpha beta gamma delta k_ss h_ss c_ss logz k1
    T = length(x)/3;
    
    k =[k1; exp(x(1:T))];
    c =[exp(x(T+1:2*T)); c_ss];
    h =[exp(x(2*T+1:3*T)); h_ss];
    
    y = exp(logz) .* k.^alpha .* h.^(1-alpha);          % implied output
    i = y - c;
end


function [trend,cycle] = hpfilter(X,lambda)
% X is a T*n matrix with n timeseries in columns
% Separate each timeseries into its trend and cycle using the HP filter
% lambda is the smoothing parameter

    T = size(X,1);
    identity = speye(T);
    secondDifference = diff(identity,2,1);
    trend = (identity + lambda*(secondDifference'*secondDifference)) \ X;
    cycle = X - trend;
end
