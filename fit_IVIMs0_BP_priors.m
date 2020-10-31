function [fr,gof,fDt,xiDt] = fit_IVIMs0_BP_priors(bvec,S,~,varargin)
% Perfom a bi-exponential fit to the DWI data of one
% voxel according to the model 
% S(b) = S_0 (f_p exp(-b D_p) + (1-f_p) exp(-b D_t)) . 
% Also computes the root-mean-square error for each fit.
% 
% Based on original code by Sebastiano Barbieri, https://github.com/sebbarb.
%
% Bayesian Probability with Markov Chain Monte Carlo Integration is used
% for the fit
%
% Usage:
%   fitBi = fitBiExponentialSegmented(S,b,varargin)
%       S: vector of DWI values corresponding to the b values specified by
%              the b vector for one image voxel.
%       b: the vector of employed b values.
%       varargin: optional vector of initial parameter estimates [Dp,Dt,Fp]
%       fr.Dp: the fitted Dp value according to the bi-exponential model.
%       fr.Dt: the fitted Dt value according to the bi-exponential model.
%       fr.fp: the fitted fp value according to the bi-exponential model.

%magnitude estimates
magDp = 0.01;
magDt = 0.001;
magFp = 0.1;
magS0 = max(S);
magP = [magDp,magDt,magFp,magS0];
%priors
priorDp = varargin{1};
priorDt = varargin{2};
priorFp = varargin{3};
priorS0 = @(s0) (s0 > 0);
prior = @(p) priorDp(p(1))*priorDt(p(2))*priorFp(p(3))*priorS0(p(4))*(p(1)>p(2));

%likelihood
% p = [Dp,Dt,Fp,s0]
N = length(S);
Q = @(p) sum( (p(4)*(p(3)*exp(-bvec*p(1)) + (1-p(3))*exp(-bvec*p(2)))-S).^2 );
likelihood = @(p) (Q(p)/2)^(-N/2);
%posterior
post = @(p) likelihood(p)*prior(p);


%MCMC
nSamples = 3000;
nKernelPoints = 1000;
if numel(varargin)>3 && ~isempty(varargin)
    initialP = [varargin{4}(1),varargin{4}(2),varargin{4}(3),varargin{4}(4)];
else
    initialP = magP;
end
samples = slicesample(initialP,nSamples,'pdf',post,'width',magP,'burnin',200);

[fDp,xiDp] = ksdensity(samples(:,1),'npoints',nKernelPoints);
[fDt,xiDt] = ksdensity(samples(:,2),'npoints',nKernelPoints);
[fFp,xiFp] = ksdensity(samples(:,3),'npoints',nKernelPoints);
[fs0,xis0] = ksdensity(samples(:,4),'npoints',nKernelPoints);
[~,idMax] = max(fDp);
tmp.Dp = xiDp(idMax);
[~,idMax] = max(fDt);
tmp.Dt = xiDt(idMax);
[~,idMax] = max(fFp);
tmp.fp = xiFp(idMax);
[~,idMax] = max(fs0);
fr.s0 = xis0(idMax);

if tmp.Dp > tmp.Dt
    fr.Dp = tmp.Dp;
    fr.Dt = tmp.Dt;
    fr.fp = tmp.fp;
else %switch values
    fr.Dp = tmp.Dt;
    fr.Dt = tmp.Dp;
    fr.fp = 1-tmp.fp;
end

fit = fr.s0*(fr.fp*exp(-bvec*fr.Dp) + (1-fr.fp)*exp(-bvec*fr.Dt));

gof.sse = sum((fit-S).^2);
gof.dfe = length(bvec)-3-1;
gof.rmse = sqrt(gof.sse/gof.dfe);
gof.rsquare = 1 - gof.sse / sum((mean(S)-S).^2);
gof.adjrsquare = 1 - (gof.sse/gof.dfe) / (sum((mean(S)-S).^2)/(length(bvec)-1));

if fr.Dt <= 0 || fr.Dt >= 1 || fr.fp < 0 || fr.fp > 1
    error('fit failed');
end

end