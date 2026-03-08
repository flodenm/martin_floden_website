%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Code to find impulse-responses for core RBC model as presented
% in lecture notes 4
%
% Also solves some alternatives and exercises at the end of LN4
%
% Martin Flodén, Fall 2025
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all; close ALL
 
prompt = "Choose:\n" + ...
"1: Show IR to TFP shock as in Figure 1 in LN4\n" + ...
"2: Show IR to TFP shock when gamma = 2, as in Figure 2\n" + ... 
"3: Show IR to TFP shock when rhoz = 0\n" + ...
"4: Show IR with k1 = 0.5*k_ss, as in Exercise 1\n" + ...
"5: Show IR to gvt spending shock as in Exercise 2\n" + ...
"6: Show IR to demand shock as in Exercise 3\n";

VERSION = input(prompt);

global alpha beta gamma delta k_ss h_ss c_ss y_ss logz G logx k1

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Calibrate the model
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
alpha = 0.4;                                        % capital share
delta = 0.02;                                       % depreciation rate of capital
if VERSION == 2
    gamma = 0.5;
else
    gamma = 2;                                      % parameter in the utility function
end
ky_target = 12;                                     % target for capital-output ratio
beta = ky_target / (alpha + (1-delta)*ky_target);   % discount factor
rhoz = 0.95;                                        % persistence of TFP shocks
rhoG = 0.95;                                        % persistence of gvt spending shocks
rhox = 0.95;                                        % persistence of demand shocks

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
r_ss = alpha*y_ss/k_ss - delta;                     % interest rate
w_ss = (1-alpha)*y_ss/h_ss;                         % wage rate


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Calculate impulse-response to
% a TFP shock
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
T = 300;                                            % assume that the economy is back in steady state T periods after shock
logz = zeros(T+1,1);
G = zeros(T+1,1);
logx = zeros(T+1,1);
k1 = k_ss;                                          % suppose capital stock = k_ss in t=1

if VERSION == 1 | VERSION == 2
    logz = rhoz.^(0:T)'*0.01;                       % TFP increases by one percent in t=1
elseif VERSION == 3
    logz(1) = 0.01;                                 % TFP shock is not persistent
elseif VERSION == 4
    k1 = 0.5*k_ss;                                  % capital stock = 0.5*k_ss in t=1
elseif VERSION == 5
    G = rhoG.^(0:T)'*0.01*y_ss;                     % G increases by one percent of ss output in t=1
elseif VERSION == 6
    logx = rhox.^(0:T)'*0.01;                       % x ("demand") increases by one percent in t=1
else
    error("Incorrect value for VERSION")
end

% By letting the solver work with the logarithm of 
% k,c and h, we ensure that k, c, and h are always
% positive
xguess = log([k_ss*ones(T,1); ...
              c_ss*ones(T,1); ...
              h_ss*ones(T,1)]);                     % guess the solution for log of k(2):k(T+1), c(1):c(T), h(1):h(T)
                        
warning off
options = optimset('Display','final');
[x,f] = fsolve(@evaluate_equilibrium_conditions,xguess,options); % find the solution in response to the TFP shock!
warning on


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Report the impulse-response to 
% the TFP shock
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[k,c,h,y,i,r,w,cg] = unpack(x);
irT = 80;                                               % number of periods to report in impulse-response graph

set(groot, 'defaultAxesFontSize', 10);
set(groot, 'defaultAxesLineWidth', .7);
set(groot, 'defaultAxesLabelFontSizeMultiplier', 1);
set(groot, 'defaultAxesTitleFontSizeMultiplier', 1);
set(groot, 'defaultFigureColor', [1 1 1]);
set(groot, 'defaultLineLineWidth', 2);    
set(groot, 'defaultLineColor', [0 71 139]/255);

scale = 100;
subplot(3,3,1); plot(1:irT,scale*(y(1:irT)/y_ss-1));
title('output'); ylabel('% deviation from ss')
subplot(3,3,2); plot(1:irT,scale*(c(1:irT)/c_ss-1));
title('consumption')
subplot(3,3,3); plot(1:irT,scale*(h(1:irT)/h_ss-1));
title('hours')
subplot(3,3,5); plot(1:irT,scale*(i(1:irT)/i_ss-1));
title('investment');
subplot(3,3,6); plot(1:irT,scale*(k(1:irT)/k_ss-1));
title('capital'); 
subplot(3,3,7); plot(1:irT,scale*4*(r(1:irT)-r_ss));     
title('interest rate'); ylabel('ppt deviation from ss')
subplot(3,3,8); plot(1:irT,scale*(w(1:irT)/w_ss-1));
title('wage'); ylabel('% deviation from ss')
subplot(3,3,9); plot(1:irT,scale*(cg(1:irT)/c_ss-1));
title('private + gvt consumption'); 

subplot(3,3,4);
if VERSION <= 4
    plot(1:irT,scale*(exp(logz(1:irT))-1)); 
    title('TFP'); ylabel('% deviation from ss')
elseif VERSION == 5
    plot(1:irT,100*G(1:irT)./y(1:irT)); 
    title('Gvt consumption'); ylabel('% of output')
    subplot(3,3,5); ylabel('% deviation from ss')
elseif VERSION == 6
    plot(1:irT,scale*(exp(logx(1:irT))-1)); 
    title('Demand'); ylabel('% deviation from ss')
end

for j = 1:9
    subplot(3,3,j); hold on;
    plot(1:irT,zeros(irT,1),'LineWidth',1,'Color',[0 0 0]);
    if j >= 7; xlabel('quarters'); end
end
set(gcf, 'Units', 'inches', 'Position', [1 1 6 4])
exportgraphics(gcf, 'rbc_ir.png', 'Resolution', 300)    % save graph to file
reset(groot);                                           % restore graphic settings to default


function f = evaluate_equilibrium_conditions(x)
% x(1:T) is guess for log k(2),..., log k(T+1)
% x(T+1:2*T) is guess for log c(1),..., log c(T)
% x(2*T+1:3*T) is guess for log h(1),..., log h(T)

    global alpha beta gamma delta k_ss h_ss c_ss y_ss logz G logx k1

    T = length(x)/3;
    
    [k,c,h,y,i] = unpack(x);
    MPK = alpha * y./k;
    MPL = (1-alpha) * y./h;
    D = exp(logx);                                      % demand
    
    evalK = k(2:end) - (1-delta)*k(1:end-1) - i(1:T);   % dynamics of the capital stock
    evalC = (D(1:end-1).*c(2:end))./(D(2:end).*c(1:end-1)) - ...
            beta*(1-delta+MPK(2:end));                  % Euler equation
    evalH = MPL.*D./c - h.^gamma;                       % optimal hours
    
    f = [evalK; evalC; evalH];

end


function [k,c,h,y,i,r,w,totalC] = unpack(x)
% x(1:T) is log k(2),..., log k(T+1)
% x(T+1:2*T) is log c(1),..., log c(T)
% x(2*T+1:3*T) is log h(1),..., log h(T)

    global alpha beta gamma delta k_ss h_ss c_ss y_ss logz G logx k1
    T = length(x)/3;
    
    k =[k1; exp(x(1:T))];
    c =[exp(x(T+1:2*T)); c_ss];
    h =[exp(x(2*T+1:3*T)); h_ss];
    
    y = exp(logz) .* k.^alpha .* h.^(1-alpha);         % implied output
    i = y - c - G;
    MPK = alpha*y./k;
    MPL = (1-alpha)*y./h;
    r = MPK - delta;
    w = MPL;
    totalC = c + G;                                    % private + government consumption
end