%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Code to find impulse-responses for core RBC model as presented
% in lecture notes 4
%
% Martin Flodén, Fall 2025
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all

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
sigma = 0.01;                                       % std deviation of TFP shocks

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
logz = rho.^(0:T)'*sigma;                           % suppose TFP increases by one std deviation in t=1
k1 = k_ss;                                          % suppose capital stock = k_ss in t=1

% By letting the solver work with the logarithm of 
% k,c and h, we ensure that k, c, and h are always
% positive
xguess = log([k_ss*ones(T,1); ...
              c_ss*ones(T,1); ...
              h_ss*ones(T,1)]);                     % guess the solution for log of k(2):k(T+1), c(1):c(T), h(1):h(T)
                        
options = optimset('Display','iter');
[x,f] = fsolve(@evaluate_equilibrium_conditions,xguess,options); % find the solution in response to the TFP shock!


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Report the impulse-response to 
% the TFP shock
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[k,c,h,y,i] = unpack(x);
irT = 150;                                               % number of periods to report in impulse-response graph

set(groot, 'defaultLineLineWidth', 2);    

subplot(2,3,1); plot(1:irT,100*(y(1:irT)/y_ss-1));
title('output'); ylabel('% deviation from ss')
subplot(2,3,2); plot(1:irT,100*(c(1:irT)/c_ss-1));
title('consumption')
subplot(2,3,3); plot(1:irT,100*(h(1:irT)/h_ss-1));
title('hours')
subplot(2,3,4); plot(1:irT,100*(exp(logz(1:irT))-1));   % assumes that z_ss = 1
title('TFP'); ylabel('% deviation from ss')
subplot(2,3,5); plot(1:irT,100*(i(1:irT)/i_ss-1));
title('investment')
subplot(2,3,6); plot(1:irT,100*(k(1:irT)/k_ss-1));
title('capital')

for j = 1:6
    subplot(2,3,j); hold on;
    plot(1:irT,zeros(irT,1),'LineWidth',1,'Color',[0 0 0]);
    plot(1:irT,100*(exp(logz(1:irT))-1),'LineWidth',1,'Color',[120 120 120]/255,'LineStyle','--');
end


function f = evaluate_equilibrium_conditions(x)
% x(1:T) is guess for log k(2),..., log k(T+1)
% x(T+1:2*T) is guess for log c(1),..., log c(T)
% x(2*T+1:3*T) is guess for log h(1),..., log h(T)

    global alpha beta gamma delta k_ss h_ss c_ss logz k1
    T = length(x)/3;
    
    [k,c,h,y] = unpack(x);
    r = alpha * y./k - delta;                           % implied interest rate
    w = (1-alpha) * y./h;                               % implied wage rate
    
    evalK = k(2:end) - (1-delta)*k(1:end-1) ...
            - (y(1:T)-c(1:T));                          % dynamics of the capital stock
    evalC = (c(2:end)./c(1:end-1)) - beta*(1+r(2:end)); % Euler equation
    evalH = w./c - h.^gamma;                            % optimal hours
    
    f = [evalK; evalC; evalH];

end


function [k,c,h,y,i] = unpack(x)
% x(1:T) is log k(2),..., log k(T+1)
% x(T+1:2*T) is log c(1),..., log c(T)
% x(2*T+1:3*T) is log h(1),..., log h(T)

    global alpha beta gamma delta k_ss h_ss c_ss logz k1
    T = length(x)/3;
    
    k =[k1; exp(x(1:T))];
    c =[exp(x(T+1:2*T)); c_ss];
    h =[exp(x(2*T+1:3*T)); h_ss];
    
    y = exp(logz) .* k.^alpha .* h.^(1-alpha);          % implied output
    i = y - c;
end