classdef LucasStokey< handle
    
    properties
        psi=.69;
        beta=.9;
        sSize=2;
        g=[.1 .2];
        pi=[.5 .5 ; .5 .5];
        A
        bMin
        bMax
        bGridSize=15;
        b_Grid
        options
        tax

    end
    
    
    methods
        function obj=LucasStokey(psi,beta,g,pi)
            
        if nargin>0    
        obj.psi=psi;
        obj.beta=beta;
        obj.sSize=length(g);
        obj.g=g;
        obj.pi=pi;
        end
        
            for s=1:obj.sSize
                for sprime=1:obj.sSize
                    if ~(s==sprime)
                        obj.A(s,sprime)=-obj.beta*obj.pi(s,sprime);
                    else
                        obj.A(s,sprime)=1-obj.beta*obj.pi(s,sprime);
                    end
                end
            end
            obj.options=optimset('Display','off');
            obj.tax=@(n,s) 1-((1-obj.psi)/(obj.psi)).*(n-obj.g(s))./(1-n);

        end
        function [c_FB,n_FB,x_FB,b_FB] = compute_FB(obj)
            %% Compute the FB
            
            c_FB=obj.psi*(1-obj.g);
            n_FB=c_FB+obj.g;
            %%
            % Solve the x associated with the FB allocation using the recursive
            % implemntability consition
            %%
            LHS=obj.psi-(1-obj.psi).*(n_FB./(1-n_FB));
            x_FB=obj.A\LHS';
            %%
            % Now use the time 0 budget constraint to get the level of assets required
            % to support the FB
            %%
            b_FB=x_FB'.*c_FB./obj.psi;
            
            
        end
        function build_grid(obj)
            [c_FB,n_FB,x_FB,b_FB] = obj.compute_FB();
            obj.bMin=min(b_FB)*1.1;
            obj.bMax=-obj.bMin;
            obj.b_Grid=linspace(obj.bMin,obj.bMax,obj.bGridSize);
            
        end
        function LSAllocation=solve_LS(obj,s0,b_,LSAllocation0)
            
            %% Solve the allocation for a range of b_,s0=1
            % Initial Guess : LSAllocation = [n0 n(s) phi]
            if nargin==3
                          
            [c_FB,n_FB,x_FB,b_FB] = compute_FB(obj);
            LSAllocation0=[n_FB(s0) n_FB 0];            
            end
            LSAllocation=fsolve(@(z) obj.ResFOC(z,b_,s0),LSAllocation0,obj.options);
        end
        function LSAllocation=solve_LS_on_grid(obj,s0)
            s0=1;
            
            [c_FB,n_FB,x_FB,b_FB] = compute_FB(obj);
            LSAllocation0=[n_FB(s0) n_FB 0];
            
            for bind=1:obj.bGridSize
                b_=obj.b_Grid(bind);
                LSAllocation(bind,:)=obj.solve_LS(s0,b_,LSAllocation0);
                LSAllocation0=LSAllocation(bind,:);
            end
           obj.tax=@(n,s) 1-((1-obj.psi)/(obj.psi)).*(n-obj.g(s))./(1-n); 
        end
        function generate_plots(obj,s0,LSAllocation)
            
            [c_FB,n_FB,x_FB,b_FB] = compute_FB(obj);
            
            %% Time 1 Assets
            for bind=1:obj.bGridSize
                S=obj.sSize;
                % Retrive the solution
                n0=LSAllocation(bind,1);
                n=LSAllocation(bind,2:2+S-1);
                phi=LSAllocation(bind,end);
                c0=n0-obj.g(s0);
                l0=1-n0;
                uc0=obj.psi/c0;
                ul0=(1-obj.psi)/l0;
                ucc0=-obj.psi/c0^2;
                ull0=-(1-obj.psi)/l0^2;
                c=n-obj.g;
                l=1-n;
                uc=obj.psi./c;
                ul=(1-obj.psi)./l;
                ucc=-obj.psi./(c.^2);
                ull=-(1-obj.psi)./(l.^2);
                % compute x from the time -1 implemntability
                LHS=uc.*c-ul.*n;
                x=obj.A\LHS';
                b(bind,:)=x'./uc;
                ArrowSec0(bind,:)=obj.beta.*obj.pi(s0,:).*uc/uc0;
                ArrowSec_l(bind,:)=obj.beta.*obj.pi(s0,:).*uc/uc(1);
                ArrowSec_2(bind,:)=obj.beta.*obj.pi(s0,:).*uc/uc(2);
                Q0(bind,:)=sum(ArrowSec0(bind,:));
                Q1(bind,:)=[sum(ArrowSec_l(bind,:)) sum(ArrowSec_2(bind,:)) ];
                
            end
            
            %% Plots
            % The red line depicts the minumum level of assets necessary to support the
            % FB. The dotted black line refers to s=1 or the low goverment expenditure
            % shock
            
            
            %% Allocation
            figure()
            subplot(2,2,1)
            plot(obj.b_Grid,LSAllocation(:,1)-obj.g(s0),'b','LineWidth',3)
            xlabel('$\mathbf b_{0}$','Interpreter','Latex','FontSize',14,'fontweight','bold')
            vline([min(b_FB)],':r')
            title('Consumption','FontSize',14,'fontweight','bold')
            hold on
            plot(obj.b_Grid,LSAllocation(:,2)-obj.g(1),'-.k','LineWidth',3)
            hold on
            plot(obj.b_Grid,LSAllocation(:,3)-obj.g(2),'k','LineWidth',3)
            xlabel('$b_{0}$','Interpreter','Latex','FontSize',14,'fontweight','bold')
            ylabel('$c(s)$','Interpreter','Latex','FontSize',14,'fontweight','bold')
            vline([min(b_FB)],':r')
            legend('c_0','c_1(g_l)','c_1(g_h)')
            
            subplot(2,2,2)
            plot(obj.b_Grid,LSAllocation(:,1),'b','LineWidth',3)
            xlabel('$b_{0}$','Interpreter','Latex','FontSize',14,'fontweight','bold')
            vline([min(b_FB)],':r')
            title('Labor supply','FontSize',14,'fontweight','bold')
            hold on
            
            plot(obj.b_Grid,LSAllocation(:,2),'-.k','LineWidth',3)
            hold on
            plot(obj.b_Grid,LSAllocation(:,3),'k','LineWidth',3)
            
            xlabel('$b_{0}$','Interpreter','Latex','FontSize',14,'fontweight','bold')
            ylabel('$n(s)$','Interpreter','Latex','FontSize',14,'fontweight','bold')
            vline([min(b_FB)],':r')
            legend('n_0','n_1(g_l)','n_1(g_h)')
            
            
            
            subplot(2,2,3)
            
            plot(obj.b_Grid,b(:,1),'-.k','LineWidth',3)
            hold on
            plot(obj.b_Grid,b(:,2),'k','LineWidth',3)
            
            xlabel('$b_{0}$','Interpreter','Latex','FontSize',14,'fontweight','bold')
            ylabel('$b(s)$','Interpreter','Latex','FontSize',14,'fontweight','bold')
            vline([min(b_FB)],':r')
            title('Time 1 assets','FontSize',14,'fontweight','bold')
            legend('b(g_l)','b_(g_h)')
            
            subplot(2,2,4)
            plot(obj.b_Grid,-LSAllocation(:,4),'k','LineWidth',3)
            xlabel('$b_{0}$','Interpreter','Latex','FontSize',14,'fontweight','bold')
            ylabel('$\phi$','Interpreter','Latex','FontSize',14,'fontweight','bold')
            vline([min(b_FB)],':r')
            title('Implementability Multiplier','FontSize',14,'fontweight','bold')
            
            
            %set(gca,'FontSize',14,'fontweight','bold')

            
            figure()
            
            %% TIME 0 labor tax
            subplot(2,2,1)
            plot(obj.b_Grid,obj.tax(LSAllocation(:,1),s0),'b','LineWidth',3)
            xlabel('$b_{0}$','Interpreter','Latex','FontSize',14,'fontweight','bold')
            ylabel('$\tau(s0)$','Interpreter','Latex','FontSize',14,'fontweight','bold')
            vline([min(b_FB)],':r')
            hold on
            plot(obj.b_Grid,obj.tax(LSAllocation(:,2),1),'-.k','LineWidth',3)
            hold on
            plot(obj.b_Grid,obj.tax(LSAllocation(:,3),2),'k','LineWidth',3)
            xlabel('$b_{0}$','Interpreter','Latex','FontSize',14,'fontweight','bold')
            ylabel('$\tau(s)$','Interpreter','Latex','FontSize',14,'fontweight','bold')
            vline([min(b_FB)],':r')
            %legend('\tau_0,' , '\tau_1(g_l)', '$\tau_2(g_h)$','Interpreter','Latex')
            title('labor taxes','FontSize',14,'fontweight','bold')
            legend('\tau_0','\tau_1(g_l)','\tau_1(g_h)')
            
            subplot(2,2,2)
            plot(obj.b_Grid,Q0,'b','LineWidth',3)
            hold on
            plot(obj.b_Grid,Q1(:,1),'-.k','LineWidth',3)
            hold on
            plot(obj.b_Grid,Q1(:,2),'k','LineWidth',3)
            legend('$Q_0(s0)$','$Q_1(s_l)$','$Q_1(s_h)$')
            h = legend;
            set(h, 'interpreter', 'latex')
            
            title('One period bonds','FontSize',14,'fontweight','bold')
            xlabel('$b_{0}$','Interpreter','Latex','FontSize',14,'fontweight','bold')
            ylabel('$Q(s)$','Interpreter','Latex','FontSize',14,'fontweight','bold')
            
            
            subplot(2,2,3)
            
            plot(obj.b_Grid,ArrowSec0(:,1),'-.k','LineWidth',3)
            hold on
            plot(obj.b_Grid,ArrowSec0(:,2),'k','LineWidth',3)
            legend('$\rho_0(s_l| s0)$','$\rho(s_h|s0)$')
            h = legend;
            set(h, 'interpreter', 'latex')
            title('Pricing Kernel - time 0','FontSize',14,'fontweight','bold')
            xlabel('$b_{0}$','Interpreter','Latex','FontSize',14,'fontweight','bold')
            ylabel('$\rho(s_1|s_0)$','Interpreter','Latex','FontSize',14,'fontweight','bold')
            
            subplot(2,2,4)
            plot(obj.b_Grid,ArrowSec_l(:,2),'-.k','LineWidth',3)
            hold on
            plot(obj.b_Grid,ArrowSec_l(:,1),'r','LineWidth',3)
            hold on
            plot(obj.b_Grid,ArrowSec_2(:,1),'k','LineWidth',3)
            legend('$\rho_1(s_h| s_l)$','$\rho_1(s|s)$','$\rho_1(s_l|s_h)$')
            h = legend;
            set(h, 'interpreter', 'latex')
            title('Pricing Kernel - Time 1','FontSize',14,'fontweight','bold')
            xlabel('$b_{0}$','Interpreter','Latex','FontSize',14,'fontweight','bold')
            ylabel('$\rho(s_2|s_1)$','Interpreter','Latex','FontSize',14,'fontweight','bold')
            %set(gca,'FontSize',14,'fontweight','bold')

            
        end    
        function [ res ] = ResFOC(obj,LSAllocation,b_,s0)
            % This function computes the residual for a given LS allocation :
            % n0,n(s),phi
            
            S=obj.sSize;
            % Retrive the guessed solution
            n0=LSAllocation(1);
            n=LSAllocation(2:2+S-1);
            phi=LSAllocation(end);
            
            c0=n0-obj.g(s0);
            l0=1-n0;
            uc0=obj.psi/c0;
            ul0=(1-obj.psi)/l0;
            ucc0=-obj.psi/c0^2;
            ull0=-(1-obj.psi)/l0^2;
            
            
            c=n-obj.g;
            l=1-n;
            uc=obj.psi./c;
            ul=(1-obj.psi)./l;
            ucc=-obj.psi./(c.^2);
            ull=-(1-obj.psi)./(l.^2);
            
            
            % compute x from the time -1 implemntability
            LHS=uc.*c-ul.*n;
            x=obj.A\LHS';
            
            
            if and(((n0>obj.g(s0)) && (n0<1)),(n>obj.g) & (n<1))
                
                % Time 0 FOC
                res(1)=(uc0-ul0)-phi*( uc0+ c0*ucc0-ul0-n0*(-ull0)-ucc0*b_ );
                % Time 1 FOC
                res(2:2+S-1)=(uc-ul)-phi*(uc+(n-obj.g).*ucc-ul-n.*(-ull));
                % Time 0 budget
                res(2+S)=uc0*c0+ obj.beta*obj.pi(s0,:)*x-ul0*n0-uc0*b_;
            else
                res= [ abs(n0-obj.g(s0)) abs(n-obj.g) phi]*100+10;
            end
            
        end     
        function simulate_shock_history(obj,b_,shock_hist)
            s0=shock_hist(1);
            b_hist(1)=b_;
            S=obj.sSize;
            LSAllocation=obj.solve_LS(s0,b_);
            tau_hist(1)=obj.tax(LSAllocation(1),s0);
            n0=LSAllocation(1);
            n=LSAllocation(2:2+S-1);
            phi=LSAllocation(end);
            c0=n0-obj.g(s0);
            l0=1-n0;
            uc0=obj.psi/c0;
            ul0=(1-obj.psi)/l0;
            ucc0=-obj.psi/c0^2;
            ull0=-(1-obj.psi)/l0^2;
            c=n-obj.g;
            l=1-n;
            uc=obj.psi./c;
            ul=(1-obj.psi)./l;
            ucc=-obj.psi./(c.^2);
            ull=-(1-obj.psi)./(l.^2);
            % compute x from the time -1 implemntability
            LHS=uc.*c-ul.*n;
            x=obj.A\LHS';
            b=x'./uc;
            for t=2:length(shock_hist)
                
                if shock_hist(t)==1
                    tau_hist(t)=obj.tax(LSAllocation(2),shock_hist(t));
                    b_hist(t)=b(1);
                else
                    tau_hist(t)=obj.tax(LSAllocation(3),shock_hist(t));
                    b_hist(t)=b(2);
                end
                
            end
            Time=linspace(0,length(shock_hist)-1,length(shock_hist));
            figure()
            subplot(3,1,1)
            plot(Time,obj.g(shock_hist),'k','linewidth',3)
            ylabel('expenditure','fontsize',14,'fontweight','bold')
            
            subplot(3,1,2)
            plot(Time,tau_hist,'k','linewidth',3)
            ylabel('taxes','fontsize',14,'fontweight','bold')
            
            subplot(3,1,3)
            plot(Time,b_hist,'k','linewidth',3)
            ylabel('assets','fontsize',14,'fontweight','bold')
            

            
        end
    end
    
end

