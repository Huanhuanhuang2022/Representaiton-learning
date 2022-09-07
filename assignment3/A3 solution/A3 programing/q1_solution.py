import math
import numpy as np
import torch


def log_likelihood_bernoulli(mu, target):
    """ 
    COMPLETE ME. DONT MODIFY THE PARAMETERS OF THE FUNCTION. Otherwise, tests might fail.

    *** note. ***

    :param mu: (FloatTensor) - shape: (batch_size x input_size) - The mean of Bernoulli random variables p(x=1).
    :param target: (FloatTensor) - shape: (batch_size x input_size) - Target samples (binary values).
    :return: (FloatTensor) - shape: (batch_size,) - log-likelihood of target samples on the Bernoulli random variables.
    """
    # init
    batch_size = mu.size(0)
    mu = mu.view(batch_size, -1)
    target = target.view(batch_size, -1)
    # print(mu)
    # print(target)

    # log_likelihood_bernoulli    

    log_prob = -torch.functional.F.binary_cross_entropy(mu, target, reduction='none')
    log_prob = log_prob.sum(1)
    print(log_prob)
    return log_prob

def log_likelihood_normal(mu, logvar, z):
    """ 
    COMPLETE ME. DONT MODIFY THE PARAMETERS OF THE FUNCTION. Otherwise, tests might fail.

    *** note. ***

    :param mu: (FloatTensor) - shape: (batch_size x input_size) - The mean of Normal distributions.
    :param logvar: (FloatTensor) - shape: (batch_size x input_size) - The log variance of Normal distributions.
    :param z: (FloatTensor) - shape: (batch_size x input_size) - Target samples.
    :return: (FloatTensor) - shape: (batch_size,) - log probability of the sames on the given Normal distributions.
    """
    # init
    batch_size = mu.size(0)
    mu = mu.view(batch_size, -1)
    logvar = logvar.view(batch_size, -1)
    z = z.view(batch_size, -1)
    # print(z)
    # print(mu)
    # print(logvar)

    # log normal
    var = torch.exp(logvar)
    log_prob = -0.5 * ((
        (mu - z)**2) / var + logvar + torch.tensor(2 * math.pi).log())
    log_prob = log_prob.sum((1))
    return log_prob


def log_mean_exp(y):
    """ 
    COMPLETE ME. DONT MODIFY THE PARAMETERS OF THE FUNCTION. Otherwise, tests might fail.

    *** note. ***

    :param y: (FloatTensor) - shape: (batch_size x sample_size) - Values to be evaluated for log_mean_exp. For example log proababilies
    :return: (FloatTensor) - shape: (batch_size,) - Output for log_mean_exp.
    """
    # init
    batch_size = y.size(0)#10
    sample_size = y.size(1)#5
    print(y)
    # log_mean_exp
    # log=[]
    # for i in range(batch_size):#k
    #     a=torch.max(y[i])
    #     exp=torch.exp(y[i]-a)
    #     log.append(exp)
    # logs = torch.stack(log,0)
    # print(logs)
    # total=logs.sum(1)
    # print(total)
    # log_mean_exp=(total/sample_size+a).log()
    # print(log_mean_exp) 
    a=torch.max(y,1).values
    aa=a.reshape(-1,1).squeeze(0)
    # print(f'aa',aa)
    sub=torch.sub(y,aa.expand(batch_size,sample_size),alpha=1)
    # print(f'sub',sub)
    exp=torch.exp(sub)
    # print(exp)
    total=exp.sum(1)
    # print(f'total',total)
    log_mean_exp=(total/sample_size).log()+a
    # print(log_mean_exp)    
    return log_mean_exp


def kl_gaussian_gaussian_analytic(mu_q, logvar_q, mu_p, logvar_p):
    """ 
    COMPLETE ME. DONT MODIFY THE PARAMETERS OF THE FUNCTION. Otherwise, tests might fail.

    *** note. ***

    :param mu_q: (FloatTensor) - shape: (batch_size x input_size) - The mean of first distributions (Normal distributions).
    :param logvar_q: (FloatTensor) - shape: (batch_size x input_size) - The log variance of first distributions (Normal distributions).
    :param mu_p: (FloatTensor) - shape: (batch_size x input_size) - The mean of second distributions (Normal distributions).
    :param logvar_p: (FloatTensor) - shape: (batch_size x input_size) - The log variance of second distributions (Normal distributions).
    :return: (FloatTensor) - shape: (batch_size,) - kl-divergence of KL(q||p).
    """
    # init
    batch_size = mu_q.size(0)
    mu_q = mu_q.view(batch_size, -1)
    logvar_q = logvar_q.view(batch_size, -1)
    mu_p = mu_p.view(batch_size, -1)
    logvar_p = logvar_p.view(batch_size, -1)

    # kld
    cov_q=torch.exp(logvar_q)
    cov_p=torch.exp(logvar_p)
    covq = torch.stack([torch.diag(var) for var in cov_q])
    # print(covq)
    covp = torch.stack([torch.diag(var) for var in cov_p])
    mvn_p = torch.distributions.multivariate_normal.MultivariateNormal(mu_q, covq)
    # print(mvn_p)
    mvn_q = torch.distributions.multivariate_normal.MultivariateNormal(mu_p, covp)
    kl=torch.distributions.kl.kl_divergence(mvn_p, mvn_q)
    # print(kl)
    return kl


# def kl_gaussian_gaussian_mc(mu_q, logvar_q, mu_p, logvar_p, num_samples=1):
#     """ 
#     COMPLETE ME. DONT MODIFY THE PARAMETERS OF THE FUNCTION. Otherwise, tests might fail.

#     *** note. ***

#     :param mu_q: (FloatTensor) - shape: (batch_size x input_size) - The mean of first distributions (Normal distributions).
#     :param logvar_q: (FloatTensor) - shape: (batch_size x input_size) - The log variance of first distributions (Normal distributions).
#     :param mu_p: (FloatTensor) - shape: (batch_size x input_size) - The mean of second distributions (Normal distributions).
#     :param logvar_p: (FloatTensor) - shape: (batch_size x input_size) - The log variance of second distributions (Normal distributions).
#     :param num_samples: (int) - shape: () - The number of sample for Monte Carlo estimate for KL-divergence
#     :return: (FloatTensor) - shape: (batch_size,) - kl-divergence of KL(q||p).
#     """
#     # init
#     # print(mu_q)
#     # print(logvar_q) 
#     # print(mu_p)
#     # print(logvar_p)
#     batch_size = mu_q.size(0)
#     input_size = np.prod(mu_q.size()[1:])
#     mu_q = mu_q.view(batch_size, -1).unsqueeze(1).expand(batch_size, num_samples, input_size)
#     logvar_q = logvar_q.view(batch_size, -1).unsqueeze(1).expand(batch_size, num_samples, input_size)
#     mu_p = mu_p.view(batch_size, -1).unsqueeze(1).expand(batch_size, num_samples, input_size)
#     logvar_p = logvar_p.view(batch_size, -1).unsqueeze(1).expand(batch_size, num_samples, input_size)

#     # kld
#     #The KL divergence is an expectation of log density ratios over distribution p. 
#     #We can approximate it with Monte Carlo samples.
#     q_sigma=torch.sqrt(torch.exp(logvar_q))
#     p_sigma=torch.sqrt(torch.exp(logvar_p))
#     cov_q=torch.exp(logvar_q)
#     cov_p=torch.exp(logvar_p)
#     covq = torch.stack([torch.diag(var) for var in cov_q])
#     covp = torch.stack([torch.diag(var) for var in cov_p])

#     #define q distribution
#     mvn_q = torch.distributions.multivariate_normal.MultivariateNormal(mu_p, covp)
#     mvn_p = torch.distributions.multivariate_normal.MultivariateNormal(mu_q, covq)
 
#     # z=mvn_q.sample(sample_shape=(num_samples,))
#     # print(z)
#     # print(z.shape)
#     # r_p = (z - mu_p) / p_sigma
#     # r_q = (z - mu_q) / q_sigma
#     # mc_kld=torch.sum(torch.log(p_sigma) - torch.log(q_sigma) + 0.5 * (r_p**2 - r_q**2), dim=-1)
#     p = mvn_q.sample(sample_shape=(num_samples,))
#     q = mvn_p.sample(sample_shape=(num_samples,))
#     # y = normpdf( x ) returns the probability density function (pdf) of 
#     # the standard normal distribution, evaluated at the values in x 
#     mc_kld=[]
#     #https://stats.stackexchange.com/questions/280885/estimate-the-kullback-leibler-kl-divergence-with-monte-carlo
#     for i in range(len(p)):
#       mc_kld.append((1.0 / len(p))*torch.sum(mvn_q.log_prob(p[i])-mvn_p.log_prob(p[i])))

#     mc_kld=torch.stack(mc_kld,0)

#     # print(mc_kld)
#     # mean = mc_kld.expanding().mean()
#     return mc_kld
def kl_gaussian_gaussian_mc(mu_q, logvar_q, mu_p, logvar_p, num_samples=1):
    """ 
    COMPLETE ME. DONT MODIFY THE PARAMETERS OF THE FUNCTION. Otherwise, tests might fail.

    *** note. ***

    :param mu_q: (FloatTensor) - shape: (batch_size x input_size) - The mean of first distributions (Normal distributions).
    :param logvar_q: (FloatTensor) - shape: (batch_size x input_size) - The log variance of first distributions (Normal distributions).
    :param mu_p: (FloatTensor) - shape: (batch_size x input_size) - The mean of second distributions (Normal distributions).
    :param logvar_p: (FloatTensor) - shape: (batch_size x input_size) - The log variance of second distributions (Normal distributions).
    :param num_samples: (int) - shape: () - The number of sample for Monte Carlo estimate for KL-divergence
    :return: (FloatTensor) - shape: (batch_size,) - kl-divergence of KL(q||p).
    """
    # init

    batch_size = mu_q.size(0)
    input_size = np.prod(mu_q.size()[1:])
    mu_q = mu_q.view(batch_size, -1).unsqueeze(1).expand(batch_size, num_samples, input_size)
    logvar_q = logvar_q.view(batch_size, -1).unsqueeze(1).expand(batch_size, num_samples, input_size)
    mu_p = mu_p.view(batch_size, -1).unsqueeze(1).expand(batch_size, num_samples, input_size)
    logvar_p = logvar_p.view(batch_size, -1).unsqueeze(1).expand(batch_size, num_samples, input_size)
    # # init
    # batch_size = mu_q.size(0)
    # mu_q = mu_q.view(batch_size, -1)
    # logvar_q = logvar_q.view(batch_size, -1)
    # mu_p = mu_p.view(batch_size, -1)
    # logvar_p = logvar_p.view(batch_size, -1)
    # print(logvar_q.shape)# 
    # # print(mu_p)
    # print(logvar_p.shape)


    # kld
    #The KL divergence is an expectation of log density ratios over distribution p. 
    #We can approximate it with Monte Carlo samples.

    cov_q=torch.exp(logvar_q)
    # print(cov_q)
    cov_p=torch.exp(logvar_p)
    # covq = torch.stack([torch.diag(var) for var in cov_q])
    # covp = torch.stack([torch.diag(var) for var in cov_p])
    covq = torch.diag_embed(cov_q)
    covp = torch.diag_embed(cov_p)
    #define q distribution
    mvn_p = torch.distributions.multivariate_normal.MultivariateNormal(mu_p, covp)
    mvn_q = torch.distributions.multivariate_normal.MultivariateNormal(mu_q, covq)
    # mvn_p = torch.distributions.Independent(torch.distributions.Normal(mu_p, cov_q), 1)
    # mvn_q = torch.distributions.Independent(torch.distributions.Normal(mu_q, cov_p), 1)
    # mvn_p = torch.distributions.multivariate_normal.MultivariateNormal(mu_p, scale_tril=covq)
    # mvn_q = torch.distributions.multivariate_normal.MultivariateNormal(mu_q, scale_tril=covp)
    # [mvn_p.batch_shape, mvn_p.event_shape]
    p = mvn_q.rsample(sample_shape=(num_samples,))
    q = mvn_p.rsample(sample_shape=(num_samples,))
    # y = normpdf( x ) returns the probability density function (pdf) of 
    # the standard normal distribution, evaluated at the values in x 
    mc_kld=[]
    #https://stats.stackexchange.com/questions/280885/estimate-the-kullback-leibler-kl-divergence-with-monte-carlo
    for i in range(len(p)):
      mc_kld.append((1.0 / len(p))*torch.sum(mvn_q.log_prob(p[i])-mvn_p.log_prob(p[i])))

    mc_kld=torch.stack(mc_kld,0)
    return mc_kld