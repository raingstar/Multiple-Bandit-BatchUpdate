class Bandit_base():
    '''Bandit base class defines the general methods needed in all bandit algorithm, like EpsilonGreedy, UCB1(Upper Confidence Bound),  
    Softmax,Thompson_Sampling, and RPM (Randomized Probability Matching).
    '''
    def __init__(self, counts, values, epsilon=0.1, alpha=1, beta=1, temperature=0.1):
        ''' self.counts: # of trials for each arm.
            self.values: average rewards for each arm. For multi binomial arm, it is the success rate for each arm.
            self.epsilon: Especially for epsilon greedy, default is 0.1
            self.BetaPrior_Alpha, self.BetaPrior_Beta: Parameters of beta prior. default is 1,1, which is uniform distribution. 
            self.temperature: Parameter for softmax
        '''
        self.counts = counts 
        self.values = values
        self.epsilon = epsilon
        self.BetaPrior_Alpha=alpha
        self.BetaPrior_Beta=beta
        self.temperature=temperature
        return
    def initialize(self, n_arms):
        '''Initialize counts and values, set all for zeros'''
        
        self.counts = [0 for col in range(n_arms)] 
        self.values = [0.0 for col in range(n_arms)] 
        return
    def set_epsilon_greedy(self, epsilon):
        '''A method to set the value of epsilon'''
        
        self.epsilon = epsilon
        return
    def set_Beta_Prior(self, alpha, beta):
        '''A method to set the value of beta prior'''

        self.BetaPrior_Alpha=alpha
        self.BetaPrior_Beta=beta
        return
    def set_temperature(self,temperature):
        self.temperature=temperature
        return
    def ind_max(self,x):
        '''Return the index of max value in list x'''
        
        m = max(x) 
        return x.index(m)
    def update(self, chosen_arm, reward): 
        '''Update the counts and values after one draw, reward will be {1,0} for bernouli trial'''
        
        self.counts[chosen_arm] = self.counts[chosen_arm] + 1 
        n = self.counts[chosen_arm]
        value = self.values[chosen_arm]
        new_value = ((n - 1) / float(n)) * value + (1 / float(n)) * reward 
        self.values[chosen_arm] = new_value
        return
    def categorical_draw(self,probs): 
        '''A simple method to generate a weighted sample, probs are the weights and sum(probs)=1'''
        
        z = random.random() 
        cum_prob = 0.0
        for i in range(len(probs)):
            prob = probs[i] 
            cum_prob += prob 
            if cum_prob > z:
                return i
        return len(probs) - 1
    def batch_update(self, arms_impression, arms_reward): 
        '''Update the counts and values after a batch, for arm1, arms_impression[i] represeants the total trials within this batch,
             while arms_reward[i] represeats the total rewards within batch(# of success for bernouli)'''
        n_arms=len(self.counts)
        for j in range(n_arms):
            n = self.counts[j]
            self.counts[j] = self.counts[j] + arms_impression[j] 
            if arms_impression[j] >0:
                self.values[j]= (n  / float(n+arms_impression[j])) * self.values[j] + (1 / float(n+arms_impression[j])) * arms_reward[j] 
        return
    def ind_top_k(self,x, k):
        '''return the index of top k out of len(x) in list x'''
        len_x=len(x)
        if k<=len_x:
            return np.argpartition(np.array(x),len_x-k)[-k:]
        else:
            random.sample(range(len_x), len_x)
class EpsilonGreedy(Bandit_base):
    '''With prob of epsilon to randomly select arms, other wise choose the arm with best values'''
    def select_arm(self):
        '''select one arm at each time'''
        if random.random() > self.epsilon:
            return self.ind_max(self.values) 
        else:
            return random.randrange(len(self.values))
    def select_k_arm(self,k):
        '''select k arms at each time'''
        if random.random() > self.epsilon:
            return self.ind_top_k(self.values,k) 
        else:
            return np.random.choice(len(self.values),min(k,len(self.values)),replace=False)
class Softmax(Bandit_base): 
    def select_arm(self):
        '''select one arm at each time'''
        z = sum([math.exp(v / self.temperature) for v in self.values]) 
        probs = [math.exp(v / self.temperature) / z for v in self.values] 
        return self.categorical_draw(probs)
    def select_k_arm(self,k):
        '''select k arms at each time'''
        z = sum([math.exp(v / self.temperature) for v in self.values]) 
        probs = [math.exp(v / self.temperature) / z for v in self.values] 
        return np.random.choice(len(self.values),min(k,len(self.values)),p=probs, replace=False)
class AnnealingSoftmax(Bandit_base):    
    def select_arm(self):
        '''select one arm at each time'''
        t = sum(self.counts) + 1
        temperature = 1 / math.log(t + 0.0000001)
        z = sum([math.exp(v / temperature) for v in self.values]) 
        probs = [math.exp(v / temperature) / z for v in self.values] 
        return self.categorical_draw(probs)
    def select_k_arm(self,k):
        '''select k arms at each time'''
        t = sum(self.counts) + 1
        temperature = 1 / math.log(t + 0.0000001)
        z = sum([math.exp(v / temperature) for v in self.values]) 
        probs = [math.exp(v / temperature) / z for v in self.values]
        return np.random.choice(len(self.values),min(k,len(self.values)),p=probs, replace=False)
class UCB1(Bandit_base):
    def select_arm(self):
        '''select one arm at each time'''
        n_arms = len(self.counts) 
        for arm in range(n_arms):
            if self.counts[arm] == 0: 
                return arm
        ucb_values = [0.0 for arm in range(n_arms)] 
        total_counts = sum(self.counts)
        for arm in range(n_arms):
            bonus = math.sqrt((2 * math.log(total_counts)) / float(self.counts[arm]))
            ucb_values[arm] = self.values[arm] + bonus 
        return self.ind_max(ucb_values)
    def select_k_arm(self,k): 
        '''select k arms at each time'''
        n_arms = len(self.counts) 
        ucb_values = [0.0 for arm in range(n_arms)] 
        total_counts = sum(self.counts)
        for arm in range(n_arms):
            bonus = math.sqrt((2 * math.log(total_counts)) / float(self.counts[arm])) if self.counts[arm]>0 else 10000
            ucb_values[arm] = self.values[arm] + bonus 
        return self.ind_top_k(ucb_values,k)
class Thompson_Sampling(Bandit_base):
    def select_arm(self):
        '''select one arm at each time'''
        n_arms = len(self.counts) 
        sucess=[int(self.counts[i]*self.values[i]+0.5) for i in range(n_arms)]
        samples=[random.betavariate(self.BetaPrior_Alpha+sucess[i],self.BetaPrior_Beta+self.counts[i]-sucess[i]) for i in range(n_arms)]
        return self.ind_max(samples)
    def select_k_arm(self,k):
        '''select k arms at each time'''
        n_arms = len(self.counts) 
        sucess=[int(self.counts[i]*self.values[i]+0.5) for i in range(n_arms)]
        samples=[random.betavariate(self.BetaPrior_Alpha+sucess[i],self.BetaPrior_Beta+self.counts[i]-sucess[i]) for i in range(n_arms)]
        return self.ind_top_k(samples,k)
class RPM(Bandit_base):
    def compute_prob(self,x, index_i):
        '''Use the formula in Scott's 2010 paper'''
        n_arms = len(self.counts) 
        sucess=[int(self.counts[i]*self.values[i]+0.5) for i in range(n_arms)]
        r=beta.pdf(x,sucess[index_i]+self.BetaPrior_Alpha, self.BetaPrior_Beta+self.counts[index_i]-sucess[index_i])
        sucess.pop(index_i)
        trials=self.counts.copy()
        trials.pop(index_i)
        for j in range(n_arms-1):
            r*=beta.cdf(x,sucess[j]+self.BetaPrior_Alpha, self.BetaPrior_Beta+self.counts[j]-sucess[j])
        return r
    def select_arm(self):
        '''select one arm at each time'''
        n_arms = len(self.counts)
        probs=[integrate.quad(lambda x: self.compute_prob(x,i),0,1)[0] for i in range(n_arms)]
        return self.categorical_draw(probs)
    def select_k_arm(self,k):
        '''select k arms at each time'''
        n_arms = len(self.counts) 
        probs=[integrate.quad(lambda x: self.compute_prob(x,i),0,1)[0] for i in range(n_arms)]
        return np.random.choice(len(self.values),min(k,len(self.values)),p=probs, replace=False)
class BernoulliArm(): 
    def __init__(self, p):
        self.p = p
    def draw(self):
        if random.random() > self.p:
            return 0.0 
        else:
            return 1.0

