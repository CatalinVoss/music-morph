import numpy as np
import matplotlib.pyplot as plt
import rewards as r

def mcmc(env, total_time, alpha, gamma, beta):
    R = lambda s: env.reward(s, display=False)
    T = lambda s,a: env.toggle(a,s)
    Ind = lambda s: env.to_onehot(s, complete=True, just_index=True)
    NUM_STATES = (2**(env.beat_types)*env.barlength)**env.num_notes
    Q = np.zeros((NUM_STATES, env.num_actions))
    def update_Q(s_i,a,r,sp_i):
        return (1-alpha)*Q[s_i,a] + alpha*(r + gamma*np.max(Q[sp_i,:]))
    rs = []
    s = env.random_state(full=True, output_onehot=False)*0 #init to zero state
    s_i = Ind(s)
    r_s = R(s)
    for t in xrange(total_time):
        rs.append(r_s)
        for a in xrange(env.num_actions):
            sp = T(s,a)
            sp_i = Ind(sp)
            r_sp = R(sp)
            Q[s_i,a] = update_Q(s_i,a,r_s,sp_i)
            if r_sp > r_s:
                s = sp
                s_i = sp_i
                r_s = r_sp
            elif np.random.random() < np.exp(beta*(r_sp - r_s)):
                s = sp
                s_i = sp_i
                r_s = r_sp
    return rs, Q


def mcmc_tester(env, beta, total_time=5000, alpha=0.5, gamma=0.999, reset_time=500):
    (rs_mc, Q) = mcmc(env, total_time=total_time, alpha=alpha, gamma=gamma, beta=beta)
    (rs_Q, Q) = env.q_learner(total_time=total_time, reset_time=None, alpha=0, gamma=gamma, Q=Q, epsilon_numerator=0)
    return np.mean(rs_mc), np.mean(rs_Q)

def test_betas(betas=np.linspace(0,10,101), time=5000, reps=16):
    midigold = np.array([[1,0,1,1]])
    music_env = r.MusicEnv(range(1),2,midigold=midigold)
    RMC = np.zeros((len(betas), reps))
    RSQ = np.zeros((len(betas), reps))
    for rep in xrange(reps):
        print rep
        for i_b, beta in enumerate(betas):
            rs_mc, rs_Q = mcmc_tester(music_env, beta, total_time=time)
            #RMC.append(rs_mc)
            RMC[i_b, rep] = rs_mc
            #RSQ.append(rs_Q)
            RSQ[i_b, rep] = rs_Q
            print i_b,
        print
    return RMC, RSQ, betas

if __name__ == "__main__":
    BETA = 0.5
    midigold = np.array([[1,0,1,1]])
    music_env = r.MusicEnv(range(1),2, midigold=midigold)
    ##(rs_mc, Q) = mcmc(music_env, total_time=5000, alpha=0.5, gamma=0.999, beta=BETA)
    ##(rs_Q, Q) = music_env.q_learner(total_time=5000, reset_time=500, alpha=0.5, gamma=0.999, Q=Q)
    betas = np.linspace(0,10,100)
    RMC = []
    RSQ = []
    for beta in betas:
        rs_mc, rs_Q = mcmc_tester(music_env, beta)
        RMC.append(rs_mc)
        RSQ.append(rs_Q)
    plt.figure()
    #plt.plot(rs_mc, label="R MC")
    #plt.hold(True)
    plt.plot(rs_Q, label="R Q")
    #plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Reward, midigold = " + str(midigold))
    plt.title("Rewards for Q learning, after MCMC exploration period of beta = " + str(BETA))

    plt.matshow(Q)
    plt.colorbar()
    plt.title("Tabular Q function")
    plt.xlabel("Action")
    plt.ylabel("State")
    plt.show()
