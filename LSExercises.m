
clear all
close all
%%

psi=.95;
beta=.9;
g=[.1 .1];
pi=[.5 .5;.5 .5];


LS=LucasStokey(psi,beta,g,pi);
LS.build_grid()
LS.generate_plots(1,LS.solve_LS_on_grid())



b0=1 % initial assets (?)
shock_hist=[1 1 1 1 2 1 1 1] % [s_0 s_1 ......s_T]
LS.simulate_shock_history(b0,shock_hist)





psi=.69
beta=.9
g=[0 0 0 0 .1 0]+.1;

% % Here is example 2
% 
pi=[0 1 0 0 0 0;
0 0 1 0 0 0;
0 0 0 1 0 0;
0 0 0 0 1 0;
0 0 0 0 0 1
0 0 0 0 0 1];

% 
% pi=[0 1 0 0 0 0;
%     0 0 1 0 0 0;
%     0 0 0 1 0 0;
%     0 0 0 0 .5 .5;
%     0 0 0 0 0 1
%     0 0 0 0 0 1];


LS_anticipated=LucasStokey(psi,beta,g,pi);

LS_anticipated.simulate_shock_history(1,[1 2 3 4 5 6 6])