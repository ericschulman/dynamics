import numpy as np
import pandas as pd
import statsmodels.api as sm
import scipy.stats as stats
import scipy.linalg as linalg
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d #pre written interpolation function
from scipy.optimize import minimize
from scipy.stats import norm

# stats
import statsmodels.api as sm
from statsmodels.base.model import GenericLikelihoodModel

from shi_test import *


#################################################

#constants/data set up
GAMMA = .5772 #euler's constant
STEP = .002


def create_states(x_obs):
    return np.arange(x_obs.min(),x_obs.max(), STEP)


def miles_pdf(i_obs, x_obs, x_next):
    """estimation of mileage pdf following AM using the
    kernel function
    
    this corresponds to pdfdx in AM's code"""
    
    #figure out max number of steps
    dx = (1-i_obs)*(x_next - x_obs) + i_obs*x_next
    
    #number of 'transition' states
    dx_states = np.arange(dx.min(),dx.max() , STEP)
    
    #use kernel groups to make pdf
    kernel1 = stats.gaussian_kde(dx, bw_method='silverman')
    pdfdx = kernel1(dx_states)
    
    return np.array([pdfdx/pdfdx.sum()]).transpose()


def transition_1(i_obs, x_obs , x_next):
    """calculate transitions probabilities,
    non-parametrically
    
    this corresponds to fmat2 in AM's code"""
    
    #states
    states = create_states(x_obs)

    #transitions when i=1
    pdfdx = miles_pdf(i_obs, x_obs, x_next).transpose()
    
    #zero probability of transitioning to large states
    zeros = np.zeros( (len(states),len(states)-pdfdx.shape[1]) )
    
    #transitioning to first state and 'jumping' dx states
    fmat1 = np.tile(pdfdx,(len(states),1))
    fmat1 = np.concatenate( (fmat1, zeros), axis=1 )

    return fmat1

def transition_0(i_obs, x_obs, x_next):
    """calculate transitions probabilities,
    non-parametrically
    
    this corresponds to fmat1 in AM's code"""
    
    states = create_states(x_obs)

    pdfdx = miles_pdf(i_obs, x_obs, x_next).transpose()
    
    #initialize fmat array, transitions when i=0
    end_zeros = np.zeros((1, len(states) - pdfdx.shape[1]))
    fmat0 = np.concatenate( (pdfdx, end_zeros), axis=1 )

    for row in range(1, len(states)):
        
        #this corresponds to colz i think
        cutoff = ( len(states) - row - pdfdx.shape[1] )
        
        #case 1 far enough from the 'end' of the matrix
        if cutoff >= 0:
            start_zeros = np.zeros((1,row))
            end_zeros = np.zeros((1, len(states) - pdfdx.shape[1] - row))
            fmat_new = np.concatenate( (start_zeros, pdfdx, end_zeros), axis=1 )
            fmat0 = np.concatenate((fmat0, fmat_new))
       
        #case 2, too far from the end and need to adjust probs
        else:
            pdf_adj = pdfdx[:,0:cutoff]
            pdf_adj = pdf_adj/pdf_adj.sum(axis=1)
            
            start_zeros = np.zeros((1,row))
            fmat_new = np.concatenate( (start_zeros, pdf_adj), axis=1 )
            fmat0 = np.concatenate((fmat0, fmat_new))
            
    return fmat0


def initial_pr(i_obs, x_obs, d=0):
    """initial the probability of view a given state following AM.
    just involves logit to predict
    
    Third arguement involves display"""
    
    states = create_states(x_obs)

    X = np.array([x_obs, x_obs**2, x_obs**3]).transpose()
    X = sm.add_constant(X)
    
    model = sm.Logit(i_obs,X)
    fit = model.fit(disp=d)
    if d: print(fit.summary())
    
    x_states = np.array([states, states**2, states**3]).transpose()
    x_states = sm.add_constant(x_states)
    
    return fit.predict(x_states)


def hm_value(params, cost, pr_obs, pr_trans, states, beta):
    """calculate value function using hotz miller approach"""
    
    #set up matrices, transition is deterministic
    trans0, trans1 = pr_trans
    #calculate value function for all state

    pr_tile = np.tile( pr_obs.reshape( len(states) ,1), (1, len(states) ))
    
    denom = (np.identity( len(states) ) - beta*(1-pr_tile)*trans0 - beta*pr_tile*trans1)
    
    numer = ( (1-pr_obs)*(cost(params, states, 0) + GAMMA - np.log(1-pr_obs)) + 
                 pr_obs*(cost(params, states, 1) + GAMMA - np.log(pr_obs) ) )
    
    value = np.linalg.inv(denom).dot(numer)
    return value


def hm_prob(params, cost, pr_obs, pr_trans, states, beta):
    """calculate psi (i.e. CCP likelihood) using 
    the value function from the hotz miller appraoch"""

    value = hm_value(params, cost, pr_obs, pr_trans, states, beta)
    value = value - value.min() #subtract out smallest value
    trans0, trans1 = pr_trans

    delta1 = np.exp( cost(params, states, 1) + beta*trans1.dot(value))
    delta0 = np.exp( cost(params, states, 0) + beta*trans0.dot(value) )
    
    return delta1/(delta1+delta0)


class CCP(GenericLikelihoodModel):
    """class for estimating the values of R and theta
    using the CCP routine and the helper functions
    above"""
    
    def __init__(self, i, x, x_next, params, cost, beta, **kwds):
        """initialize the class
        
        i - replacement decisions
        x - miles
        x_next - next periods miles
        params - names for cost function parameters
        cost - cost function specification, takes agruements (params, x, i) """
        
        super(CCP, self).__init__(i, x, **kwds)


        ##size of step in discretization
        self.beta = beta
        self.states = create_states(x)

        #data
        self.endog = i #these names don't work exactly
        self.exog = x #the idea is that x is mean indep of epsilon
        self.x_next = x_next
        
        #transitions
        self.pr_obs = initial_pr(i, x)
        self.trans =  transition_0(i,x,x_next), transition_1(i,x,x_next)
        
        #initial model fit
        self.cost = cost
        self.num_params = len(params)
        self.data.xnames =  params
        self.results = self.fit( start_params=np.ones(self.num_params) )
        
        
    def nloglikeobs(self, params, v=False):
        """psuedo log likelihood function for the CCP estimator"""
        
        # Input our data into the model
        i = self.endog
        x = (self.exog/STEP).astype(int)*STEP #discretized x
           
        #set up hm state pr
        prob = hm_prob(params, self.cost, self.pr_obs, self.trans,self.states,self.beta).transpose()
        prob = interp1d(self.states, prob)
        prob = prob(x)
        
        log_likelihood = (1-i)*np.log(1-prob) + i*np.log(prob)
        
        return -log_likelihood
    
    
    def iterate(self, numiter):
        """iterate the Hotz Miller estimation procedure 'numiter' times"""
        i = 0
        while(i < numiter):
            #update pr_obs based on parameters
            self.pr_obs = hm_prob(self.results.params, self.cost, self.pr_obs, self.trans)
            
            #refit the model
            self.results = self.fit(start_params=np.ones(self.num_params))
            i = i +1




####################################################################################


def setup_test(data):
    linear_cost = lambda params, x, i: (1-i)*x*params[i] + i*params[i]
    model1 = CCP(data['replace'], data['miles'], data['miles_next'], ['theta1','RC'], linear_cost,0)
    print(model1.results.params)
    model1_fit = model1.results
    ll1 = model1.loglikeobs(model1_fit.params)
    grad1 = model1.score_obs(model1_fit.params)
    hess1 = model1.hessian(model1_fit.params)
    params1 = model1_fit.params

    model2 = CCP(data['replace'], data['miles'], data['miles_next'], ['theta1','RC'], linear_cost,.9999)
    model2_fit = model2.results
    print(model2.results.params)
    ll2 = model2.loglikeobs(model2_fit.params)
    grad2 = model2.score_obs(model2_fit.params)
    hess2 = model2.hessian(model2_fit.params)
    params2 = model2_fit.params

    return ll1, grad1, hess1, params1, ll2, grad2, hess2, params2


def regular_test(data, setup_test):
    ll1, grad1, hess1, params1, ll2, grad2, hess2, params2 = setup_test(data)
    nobs = ll1.shape[0]
    llr = (ll1 - ll2).sum()
    omega = np.sqrt((ll1 - ll2).var())
    test_stat = llr/(omega*np.sqrt(nobs))
    print('regular: test, llr, omega ----')
    print(test_stat, llr, omega)
    print('---- ')
    return 1*(test_stat >= 1.96) + 2*(test_stat <= -1.96),test_stat


# helper functions for bootstrap

def compute_eigen2(ll1,grad1,hess1,params1,ll2,grad2,hess2,params2):
    """required for computing bias adjustement for the test"""
    n = ll1.shape[0]
    hess1 = hess1/n
    hess2 = hess2/n

    k1 = params1.shape[0]
    k2 = params2.shape[0]
    k = k1 + k2
    
    #A_hat:
    A_hat1 = np.concatenate([hess1,np.zeros((k2,k1))])
    A_hat2 = np.concatenate([np.zeros((k1,k2)),-1*hess2])
    A_hat = np.concatenate([A_hat1,A_hat2],axis=1)

    #B_hat, covariance of the score...
    B_hat =  np.concatenate([grad1,-grad2],axis=1) #might be a mistake here..
    B_hat = np.cov(B_hat.transpose())

    #compute eigenvalues for weighted chisq
    sqrt_B_hat= linalg.sqrtm(B_hat)
    W_hat = np.matmul(sqrt_B_hat,linalg.inv(A_hat))
    W_hat = np.matmul(W_hat,sqrt_B_hat)
    V,W = np.linalg.eig(W_hat)

    return V


def bootstrap_distr(ll1,grad1,hess1,params1,ll2,grad2,hess2,params2,c=0,trials=500):
    nobs = ll1.shape[0]
    
    test_stats = []
    variance_stats = []
    llr = ll1-ll2
     
    for i in range(trials):
        np.random.seed()
        sample  = np.random.choice(np.arange(0,nobs),nobs,replace=True)
        llrs = np.array(llr)[sample]
        test_stats.append( llrs.sum() )
        variance_stats.append( llrs.var() )

    #final product, bootstrap
    V =  compute_eigen2(ll1,grad1,hess1,params1,ll2,grad2,hess2,params2)
    test_stats = np.array(test_stats+ V.sum()/(2))
    variance_stats = np.sqrt(np.array(variance_stats)*nobs + c*(V*V).sum())

    #set up test stat   
    omega = np.sqrt((ll1 - ll2).var()*nobs + c*(V*V).sum())
    llr = (ll1 - ll2).sum() +V.sum()/(2)
    return test_stats,variance_stats,llr,omega

# TODO 4: Get Bootstrap test working

def bootstrap_test(data,setup_test,c=0,trials=500,alpha=.05):
    ll1,grad1,hess1,params1,ll2,grad2,hess2,params2 = setup_test(data)

    #set up bootstrap distr
    test_stats,variance_stats,llr,omega  = bootstrap_distr(ll1,grad1,hess1,params1,ll2,grad2,hess2,params2,c=c,trials=trials)
    test_stats = test_stats/variance_stats
    
    #set up confidence intervals
    cv_lower = np.percentile(test_stats, 50*alpha, axis=0)
    cv_upper = np.percentile(test_stats, 100-50*alpha, axis=0)

    return  2*(0 >= cv_upper) + 1*(0 <= cv_lower), cv_lower, cv_upper


def test_table(data,setup_test, trials=100):
    
    #bootstrap cv
    result_boot, cv_lower1, cv_upper1 = bootstrap_test(data,setup_test, trials=trials,alpha=.1)
    result_boot, cv_lower2, cv_upper2 = bootstrap_test(data,setup_test, trials=trials,alpha=.05)
    result_boot, cv_lower3, cv_upper3 = bootstrap_test(data,setup_test, trials=trials,alpha=.01)
    
    #regular result
    result_class, test_stat = regular_test(data,setup_test)
    
    #shi results
    result_shi, stat_shi1, cv_shi1= ndVuong(data,setup_test,alpha=.1)
    result_shi, stat_shi2, cv_shi2= ndVuong(data,setup_test,alpha=.05)
    result_shi, stat_shi3, cv_shi3= ndVuong(data,setup_test,alpha=.01)


    print('\\begin{center}\n\\begin{tabular}{ccccc}\n\\toprule')
    print('\\textbf{Version} & \\textbf{Result} & \\textbf{90 \\% CI} & \\textbf{95 \\% CI} & \\textbf{99 \\% CI} \\\\ \\midrule' )
    print('Shi (2015) & H%s & [%.3f, %.3f] & [%.3f, %.3f] & [%.3f, %.3f] \\\\'%(result_shi, 
                                                  stat_shi1- cv_shi1, stat_shi1+ cv_shi1,
                                                  stat_shi2- cv_shi2,stat_shi2+ cv_shi2,
                                                  stat_shi3- cv_shi3,stat_shi3+ cv_shi3))
    print('Classical & H%s & [%.3f, %.3f] & [%.3f, %.3f] & [%.3f, %.3f] \\\\'%(result_class,
                                                 test_stat- 1.645,test_stat+ 1.645,
                                                 test_stat- 1.959,test_stat+ 1.959,
                                                 test_stat- 2.576,test_stat+ 2.576))
    print('Bootstrap & H%s & [%.3f, %.3f] & [%.3f, %.3f] & [%.3f, %.3f] \\\\'%(result_boot,
                                                 cv_lower1,cv_upper1,
                                                 cv_lower2,cv_upper2,
                                                 cv_lower3,cv_upper3))
    print('\\bottomrule\n\\end{tabular}\n\\end{center}')