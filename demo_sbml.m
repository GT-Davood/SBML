clc; close all; clearvars; 
addpath('..\common','..\data','..\distRatio','..\sbml','..\lca');
rng(1);
alg = 'lca';
ds = 'wine';

%% adjust common parameters
saveflag = 1; numRun = 1; 
params.kf = 100;
params.p = 12;  
params.kn = 3; params.margin = 1;
params.normalize = 1; params.scale = 1;
MaxIns = 5000; % to speed up computing "training correct rate"
evalAlg = 'knn'; params.evalAlg = evalAlg;
kernelParams.coding = 'onevsone';  kernelParams.standardize = 1;
%% load data
[X,y,c] = loadData(ds, struct('biasFlag',0,'normalize',params.normalize));
d = size(X,1);
X = (1/params.scale)*X;
%% adjust the alg parameters
switch(lower(alg))
    case 'sbml'
        % sbml parameters
        params.maxIter1 = 3; params.maxIter = 100; 
        params.batchSize = 30000; % does not work currently
        params.p = 10;  
        params.a0 = 1000; 
        params.sigma = 1; params.gamma = 1;
        params.WInit_Method = 'pca';
        
        % rpt parameters
        params.echo = 1; % set to 1 to indicate test and train correct rate should be displayed during running of the algorithm
        params.enforceMaxIter = 1; % the learning process proceed until max iteration reached
        params.cmpPrevRes = inf;
        params.rptFlag = 1; params.rpt_interval = 50;
       
    case 'distratio'
    case 'euc'
        params.p = d;
end
switch(lower(evalAlg))
    case {'knn'}
        params.kn = 3;
    case 'svm-lin'
        kernelParams.scale = 1; kernelParams.kernel = 'linear'; 
        kernelParams.C = 10;
    case 'svm-gauss'
        kernelParams.scale = 15; kernelParams.kernel = 'gaussian';
        kernelParams.C = 10;
    case 'svm-pol'
        kernelParams.scale = 12; kernelParams.kernel = 'polynomial';
        kernelParams.polOrder = 5; kernelParams.C = .2;
end
evalAlgDesc = evalAlg;
if(regexp(evalAlg,'svm') > 0)
    evalAlgDesc = sprintf('%s%g',evalAlg,kernelParams.scale);
    if(regexp(evalAlg,'pol') > 0)
      evalAlgDesc = sprintf('%s_o%g',evalAlgDesc,kernelParams.polOrder);
    end
end
params.kernelParams = kernelParams;
%% run and eval the algorithm
corr = zeros(numRun,1); corrTr = zeros(numRun,1); 
cp = cell(numRun,1); cpTr = cell(numRun,1); 
W = cell(numRun,1); 
rpt = cell(numRun,1);

rng(1); % for reproducibility of results
for t=1:numRun
    options.meanFlag =0; options.sampleReductionFlag = 0;
    options.additive = 1;
    [XTr,yTr,XTe,yTe,~,~,trInd,teInd,valInd] = divRand(X,y,params.kf);
    [XTr,yTr,S,D] = genSD3(XTr,yTr,c,params.kn,params.margin,options);
    params.XTe = XTe; params.yTe = yTe; params.yTr = yTr;
    % run algorithm
    cprintf('*red','*************************run %d **************************\n',t);
    %% run algorithm
    switch(lower(alg))
        case 'sbml'
            cprintf('*comments','sbml %s %s f=%0.2f p=%d gamma=%g  \n',evalAlgDesc,ds,params.kf,params.p,params.gamma);
            [W{t},~,~,rpt{t}] = sbml(XTr,S,D,params.p, params);
            
        case 'distratio'
              W{t} = learn_mahalanobis_metric(XTr, S, D, params.p);
        case 'lca'
              W{t} = lca(XTr, S, D, params.p)';
        case 'euc'
             W{t} = eye(d,d);
    end
   %% eval the algorithm
    switch(lower(evalAlg))
        case 'knn'
            [corr(t),corrTr(t),cp{t},cpTr{t},yhatTe] = knn_test(W{t},XTr,yTr,XTe,yTe,params.kn,MaxIns);
        case {'svm-lin','svm-gauss','svm-pol'}
            [corr(t),corrTr(t),cp{t},cpTr{t},yhatTe] = svm_test(XTr'*W{t},yTr,XTe'*W{t},yTe,kernelParams,MaxIns);
    end
    [~,AA] = compCorrPerClass(cp(t),1,yTe);
    if(strcmp(alg,'sbml') && params.rptFlag)
        [meanRpt,bestRes,bestAA,bestResTr,bestIter1, bestIter] = compMeanRpt(rpt(t));
        cprintf('*red','%s-%s Test Correct Rate:%0.2f, Test AA:%0.2f, bestIter=(%d,%d) \n',evalAlg,alg,bestRes,bestAA, bestIter1,bestIter);
        cprintf('*blue','%s-%s Train Correct Rate:%0.2f bestIter=(%d,%d) \n',evalAlg,alg,bestResTr,bestIter1,bestIter);
    else
        cprintf('red','%s-%s Test Correct Rate:%0.2f \t\t',evalAlg,alg,corr(t));
        cprintf('blue','%s-%s Train Correct Rate:%0.2f \n',evalAlg,alg,corrTr(t));
        cprintf('Strings','%s-%s Test AA Rate:%0.2f \n',evalAlg,alg,AA);
    end
    if(t==1 || corr(t)>corr(t-1))
        gt_hat = genIndianaPredGT(y,yhatTe,teInd);
    end
end
fprintf(2,'**********************************************************\n');
meanCorr = mean(corr); meanCorrTr = mean(corrTr);
[corrPerClass,AA] = compCorrPerClass(cp,numRun,yTe);
corrPerClassStr = arrayfun(@num2str,corrPerClass,'UniformOutput', false);
%% preparing mean rpt
if(strcmp(alg,'sbml') && params.rptFlag)
    [meanRpt,bestRes,bestAA,bestResTr,bestIter1, bestIter] = compMeanRpt(rpt);
    cprintf('*red','%s-%s Test Correct Rate:%0.2f, Test AA:%0.2f, bestIter=(%d,%d) \n',evalAlg,alg,bestRes,bestAA, bestIter1,bestIter);
    cprintf('*blue','%s-%s Train Correct Rate:%0.2f bestIter=(%d,%d) \n',evalAlg,alg,bestResTr,bestIter1,bestIter);
else
    cprintf('*red','%s-%s Test Mean Correct Rate:%0.2f, Test AA:%0.2f  \t',evalAlg,alg,meanCorr,AA);
    cprintf('*blue','%s-%s Train Mean Correct Rate:%0.2f \n',evalAlg,alg,meanCorrTr);
end

%% save res
if(saveflag)
    params = rmfield(params, {'XTe','yTr','yTe'});
    switch(lower(alg))
        case 'sbml'
        fname = sprintf('.\\Rpt%s\\%s_%s%g_%0.2f_%s_%d_%g.mat',...
           ds,alg,ds,params.kf,bestRes,evalAlgDesc,params.p, params.gamma);
        save(fname,'meanCorr', 'meanCorrTr', 'corr', 'corrTr', 'cp', 'cpTr', 'params'...
            ,'numRun','W','evalAlg','kernelParams','meanRpt','rpt','bestRes',...
            'bestIter1', 'bestIter','bestAA','corrPerClass','corrPerClassStr','AA','gt_hat');
        
        otherwise
        fname = sprintf('.\\Rpt%s\\%s_%s_%g_%0.2f_%s_%d.mat',...
           ds,alg,ds,params.kf,meanCorr,evalAlgDesc, params.p);
        save(fname,'meanCorr', 'meanCorrTr', 'corr', 'corrTr','cp', 'cpTr', 'params'...
            ,'numRun','W','evalAlg','kernelParams','corrPerClass','corrPerClassStr','AA','gt_hat');
    end
end
