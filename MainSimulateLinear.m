% This script contains the code to simuate the lineraized policy rules with
% persistence
clear all
% Primitives
S=3
gamma=1
beta=.98
gamma=gamma;
beta=beta;
g=(linspace(.95,1.10,S)*.17)';
alpha=.5;
for s_=1:S
    for s=1:S
        if s_==s
        PI(s_,s)=alpha;
        else
            PI(s_,s)=(1-alpha)/(S-1);
        end
    end
end

Pvg=(eye(S)-beta*PI)^-1*g
tau=@(mu) gamma*mu/((1 +gamma)*mu - 1);

d_tau=@(mu) -gamma / ( (1+gamma)*mu - 1 )^2;    
I=@(mu) (1-tau(mu))^(1/gamma) * tau(mu);

d_I= @(mu) (1/gamma)*(1-tau(mu))^(1/gamma-1)*d_tau(mu)+ d_tau(mu)*(1-tau(mu))^(1/gamma);
mu=-.25
bbar=@(mu) beta*I(mu)/(1-beta) - beta*PI*Pvg
bbar_vec=bbar(mu);

 mean(bbar_vec)
%         
% for s_=1:S
%     for s=1:S
%         
%         ppbar(s_,s)=1-(beta./bbar_vec(s_))*(Pvg(s)-PI(s_,:)*Pvg);
%     end
% end
% 
% Epbar2=PI(1,:)*(ppbar(1,:).^2)';
% 
% 
% factor=-mu*beta*(d_I(mu)+d_b_d_mu(mu));




% Compute Coeff

%  assets - mu
d_b_d_mu=@(mu) beta*d_I(mu)./(Epbar2-beta);

%  assets - p
%  assets - alpha


%  muprime - mu
%  muprime  - p
%  muprime - alpha


% simulate 

% plot ergodic distribution



