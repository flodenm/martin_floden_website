%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% How is the Ramsey model affected by a fall in
% the discount factor? (EX5 in LN2)
%
% We start in old steady state with beta = 0.96
% Then beta suddenly falls to beta = 0.90
%
% Assume: 
% Cobb Douglas production
% Log utility
% Constant productivity (g=0, A=1)
% Constant population (n=0, L=1)
% Initial capital stock = k0
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Calibrate the model and find steady state
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
alpha = 0.4;                                            % capital share in production
oldbeta = 0.96;                                         % initial discount factor
beta = 0.9;                                             % new discount factor
delta = 0.10;                                           % depreciation rate of capital (one period = one year)

kss = (alpha*beta/(1-beta*(1-delta)))^(1/(1-alpha));    % steady state capital stock
css = kss^alpha - delta*kss;                            % steady state consumption

oldkss = (alpha*oldbeta/(1-oldbeta*(1-delta)))^(1/(1-alpha));
oldcss = oldkss^alpha - delta*oldkss;

k0  = oldkss;                                          % initial capital stock


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Find transition to steady state
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
T = 300;                                                % assume that the economy is in steady state after T periods

% By letting the solver work with the logarithm of K and C, 
% we ensure that K and C are always positive
xguess = log([kss*ones(T,1); ...
              css*ones(T,1)]);                          % guess the solution for log of k(2):k(T+1), c(1):c(T)
                        
global alpha beta delta kss css k0
opt = optimset('Display','iter','TolX',1e-12);
[x,f] = fsolve(@evaluate_equilibrium_conditions,xguess,opt); % find the solution!



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Report the solution
% Show also 10 periods before change in beta
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
k = [oldkss*ones(10,1); k0; exp(x(1:T-1))];
c = [oldcss*ones(10,1); exp(x(T+1:2*T))];

figure
subplot(1,2,1); plot(k); title('Capital'); xlabel('Time');
subplot(1,2,2); plot(c); title('Consumption'); xlabel('Time');

%%% Let us also try to reconstruct the phase diagram
figure

% first the Dk = 0 locus
kmax = delta^(1/(alpha-1));                  
kgrid = 0:(kss/100):kmax;               
cvals = kgrid.^alpha - delta*kgrid;
plot(kgrid,cvals,'Color','black','LineWidth',2); hold on
title('Phase diagram')
xlabel('Capital stock, k')
ylabel('Consumption, c')

% second the old and new Dc = 0 locus
plot([oldkss oldkss],[0 2*css],'black','LineWidth',2)
plot([kss kss],[0 2*css],'red','LineWidth',2)

% now add the (k,c) values that we solved for in the transition
% do it slowly, period by period, but just 50 steps
for t = 1:50
    plot(k(t),c(t),'Color','red','Marker','.','MarkerSize',10);
    pause(0.15);                                        % pause 0.15 seconds between periods
end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Function to evaluate equilibrium conditions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function f = evaluate_equilibrium_conditions(x)
% x(1:T) is guess for log k(2),..., log k(T+1)
% x(T+1:2*T) is guess for log c(1),..., log c(T)

    global alpha beta delta kss css k0
    T = length(x)/2;
    
    k = [k0; exp(x(1:T))];
    c = [exp(x(T+1:2*T)); css];                         

    y = k.^alpha;
    MPK = alpha * y./k;                         
    
    evalK = k(2:end) - (1-delta)*k(1:end-1) - (y(1:T)-c(1:T));      % dynamics of the capital stock
    evalC = (c(2:end)./c(1:end-1)) - beta*(1-delta+MPK(2:T+1));     % Euler equation
    
    f = [evalK; evalC];

end

