% learning
% This function implements sparse baysian metric learning method
% inputs: X: a d*N matrix where N and d represents number of instances and 
%            shows dimension of data
%         S: set of similar pairs, each row represent a pair of similar
%            instances
%         D: set of disimilar pairs
%         p:maximum dimensionality of outputs
%         a0: initial value for a
%outputs: W: a d*p mapping matrix from input to latent space
%         sigma: variance of p(z|x) distribution
%         gamma: variance of p(y=1|z,z') distribution
%---------------------------------------------------------------------
function [W_best,sigma_best,a,rpt] = sbml(X, S, D, p,params)
    global Max_Alpha Ns indRpt
    
    [a0,echo,y,maxIter1,maxIter,sigma,gamma] = parseParams(params);
    [d,N] = size(X);
    if(nargin < 5)
        a0 = 1e01;
    end
    rpt = zeros(maxIter1*maxIter/params.rpt_interval + 1,5);
    indRpt = 0;
    Max_Alpha = 10000;
    a =  a0 * ones(1,p); % a: percision vector
    switch params.WInit_Method
        case 'eye'
            W = eye(d,p);
        case 'W0'
            W = params.W0;
        otherwise
            W = pca(X','NumComponents',p);
    end
    Ns = size(S,1); Nd = size(D,1);M = Ns + Nd;
    A = zeros(M,3);
    A(1:Ns, 1:2) = S;
    A(Ns+1:end,1:2) = D;
    A(1:Ns,3) = 1; A(Ns+1:end,3) = 0;
    
    tolFun1 = 1e-03;
    tolFun = 1e-05;
    
%     Xi = X(:,A(:,1)); Xj=X(:,A(:,2));
%     SumC = Xi * Xi' + Xj*Xj';

    I = A(:,1); J = A(:,2); IS = I(1:Ns); ID = I(Ns+1:end);
    JS = J(1:Ns); JD = J(Ns+1:end);

    coeff = accumarray([I;J],1,[N,1])'; 
    Coeff = repmat(coeff,d,1);
    SumC = (Coeff.*X)*X';

    indS1 = accumarray(IS,(1:Ns)',[N 1],@(x) {x});
    indS2 = accumarray(JS,(1:Ns)',[N 1],@(x) {x});
    indD1 = accumarray(ID,(1:Nd)',[N 1],@(x) {x+Ns});
    indD2 = accumarray(JD,(1:Nd)',[N 1],@(x) {x+Ns});
    GZ = zeros(N,p); GV = zeros(N,p);

    Temp = W'*X;
    WXi = Temp(:,I); WXj = Temp(:,J);
    WX = WXi - WXj;
    WXS = WX(:,1:Ns); WXD = WX(:,Ns+1:end);
   
    [err,v,err_emp,sim,cnwx] = compute_err(W,WXS,WXD,sigma, gamma,a);
    err_min = err;
    W_best = W; sigma_best = sigma;
    
    if(echo==1)
        rpt = displayRes(0,0,err,0,err_emp,W,X,y,params,sim,cnwx,rpt);
    else
        textprogressbar('percent = ', 'init');
    end
    
%     h = waitbar(0,'Please wait...');
    for num=1:maxIter1
       for iter=1:maxIter
           if(echo == 0 && mod(iter,10) == 0)
              textprogressbar( ((num-1)*maxIter+iter)/ (num*maxIter) *100);
           end
         

%            waitbar(iter / maxIter,h,sprintf('iter=%d',iter));
            % E step
            % Z = [z1,z2,...,zM]
            t = gamma + 2*sigma;

            Z = WXi;
            Z(:,1:Ns) = Z(:,1:Ns) - (sigma/t)*WXS;
            Z(:,Ns+1:end) = Z(:,Ns+1:end) + ...
                (sigma/t)*(repmat(v,p,1).*WXD);
            
            % V = [z'1,z'2,...,z'M]
            V = WXj;
            V(:,1:Ns) = V(:,1:Ns) + (sigma/t)*WXS;
            V(:,Ns+1:end) = V(:,Ns+1:end) - ...
                (sigma/t)*(repmat(v,p,1).*WXD);
            
            % M step
            %update W
%             C = Xi*Z' + Xj*V';
            
            for i=1:N
                 GZ(i,:) = sum(Z(:,indS1{i}),2)' + sum(Z(:,indD1{i}),2)';
                 GV(i,:) = sum(V(:,indS2{i}),2)' + sum(V(:,indD2{i}),2)';
            end
            C = X*(GZ + GV);
%             C = (1/sigma)*(X*(GZ + GV)) + eye(d,p)*diag(a);
%             u = unique(a);
%             W_new = zeros(d,p);
%             for i=1:length(u)
% %                 W_new(:,a == u(i)) = (SumC+ sigma*u(i)*eye(d))\C(:,a == u(i));
%                   W_new(:,a == u(i)) = ((1/sigma)*SumC+ u(i)*eye(d))\C(:,a == u(i));
%             end
            W_new = sylvester(SumC,sigma*diag(a),C);
            Temp = W_new'*X;
            WXi = Temp(:,I); WXj = Temp(:,J);
            WX = WXi - WXj;
            WXS = WX(:,1:Ns); WXD = WX(:,Ns+1:end);
            
            % update sigma
            eps = Ns*(1-sigma/t) + sum(1+v*(sigma/t));
            eps = eps*p*sigma;
            EW = 0.5*( norm(Z - WXi,'fro')^2 + norm(V - WXj,'fro')^2 );
            sigma_new = 1/(p*M) * (EW + eps);
            [err_new,v,err_emp,sim,colNormWX] = compute_err(W_new,WXS,WXD,sigma_new, gamma,a);
            
            errRatio = abs(err - err_new)/err_min;
            if((params.enforceMaxIter == 0) && (err_new > err || errRatio < tolFun))
                break;
            else
                sigma = sigma_new;
                W = W_new;
            end 
            
            if(mod(iter,params.rpt_interval)==0 && (echo == 1 || params.rptFlag == 1))
               rpt = displayRes(num,iter,err_new,errRatio,err_emp,W,X,y,params,sim,colNormWX,rpt);
               if(params.rptFlag && indRpt > params.cmpPrevRes)
                   % if current test_correct_rate is less than from all 3
                   % previous reported test_correct_rates then break
                   curTestRate = rpt(indRpt,3);
                   prevTestRates = rpt(indRpt-params.cmpPrevRes:indRpt-1,3);
                   if(sum(curTestRate < prevTestRates) == params.cmpPrevRes) 
                       break;
                   end
               end
               
            end
            err = err_new;
       end %end for i
       
       errRatio = abs(err_min - err_new)/err_min;
       if((params.enforceMaxIter == 0) && (errRatio < tolFun1))
            break;
       end
      
       % update alpha(i)
       colnormW = sum(W .* W, 1);
       a = min(Max_Alpha,d./ colnormW);

       W_best = W;
       sigma_best = sigma;
       err_min = err_new;
       err = err_new;
       
    end % end for num
%     delete(h)       % DELETE the waitbar; don't try to CLOSE it.
    textprogressbar('finished!!!','stop');
end

function [a0,echo,y,maxIter1,maxIter,sigma,gamma] = parseParams(params)
    a0=params.a0;
    echo = params.echo;
    y = params.yTr;
    
    if(isfield(params,'maxIter1'))
        maxIter1 = params.maxIter1;
    else
        maxIter1 = 1;
    end
    if(isfield(params,'maxIter'))
        maxIter = params.maxIter;
    else
        maxIter = 1000;
    end
    
    if(isfield(params,'sigma'))
        sigma = params.sigma;
    else
        sigma = 1;
    end
    
    if(isfield(params,'gamma'))
        gamma = params.gamma;
    else
        gamma = 1;
    end
end

% compute error and odd vector v
function [err,v, err_emp,Sim,colNormWX] = compute_err(W,WXS,WXD,sigma, gamma,a)
    global Max_Alpha 
    
    % copmute v:[v1,v2,...,vNd]: odd vector and error
    [d,p] = size(W);
    t = gamma + 2*sigma;
    coeff = (gamma /t)^(p/2);
    
    colnormWXS = sum(WXS.*WXS,1);
    objVal = sum(log(coeff) - colnormWXS/(2*t));
    colnormWXD = sum(WXD .* WXD,1);
    v = coeff.* exp(-colnormWXD ./(2*t));
    objVal = objVal + sum(log(1-v));
    v = v ./ (1-v);
    err_emp = -objVal;
    
    colNormWX = [colnormWXS,colnormWXD];
    Sim = exp(log(coeff)-colNormWX /(2*t))';
    
    colNormW = sum(W.*W,1);
    ind = a ~= Max_Alpha;
    nzero = p - sum(ind);
    objVal = objVal +  0.5*(sum( d*log(a/(2*pi))) - ...          
                sum(a(ind).*colNormW(ind)) - nzero*d);
    err = -objVal;
end

function rpt = displayRes(num,iter,err,errRatio,err_emp,W,X,y,params,sim,colNormWX,rpt)
    global Ns indRpt
    MaxIns = 10000;
    
    switch(lower(params.evalAlg))
        case 'knn'
            [corr,corrTr,cp] = knn_test(W,X,y,params.XTe,params.yTe,params.kn,MaxIns);
        case {'svm-lin','svm-gauss','svm-pol'}
            [corr,corrTr,cp] = svm_test(X'*W,y,params.XTe'*W,params.yTe,...
                params.kernelParams,MaxIns);
    end
   [~,avgAcc] = compCorrPerClass({cp},1,params.yTe);

    
    if(params.rptFlag)
%         rpt= [rpt;[num,iter,corr,corrTr]];
        indRpt = indRpt +1;
        rpt(indRpt,:) = [num,iter,corr,corrTr,avgAcc];
    end

    if(params.echo == 1)
       fprintf('iter=(%d,%d), %s test corrRate=%0.2f, AA =%0.2f, train corrRate=%0.2f, err=%0.3f, err_ratio=%0.3f, err_emp=%0.3f\n',...
       num,iter,params.evalAlg,corr,avgAcc,corrTr,err,errRatio,err_emp);
       c = length(unique(y));
       options.meanFlag = 0; options.sampleReductionFlag = 1;
       [~,~,~,DD] = genSD3(W'*X,y,c,params.kn,1,options);
       n_imp = size(DD,1);
       n_active = size(unique(DD(:,1)),1);
       cprintf('*[1,0.5,0]','number of data points with active Imp:%d, number of Imp:%d\n',n_active,n_imp);
       meanSimS = mean(sim(1:Ns));  meanSimD = mean(sim(Ns+1:end));
       meanCnwxs = mean(colNormWX(1:Ns)); meanCnwxd = mean(colNormWX(Ns+1:end));
       cprintf('blue','meanSimS = %f, meanSimD = %f, ratioSim=%f\n',...
                meanSimS,meanSimD,meanSimS/meanSimD);
       cprintf('red','meanCnwxs=%f, meanCnwxd=%f\n',meanCnwxs,meanCnwxd); 
    end
end










