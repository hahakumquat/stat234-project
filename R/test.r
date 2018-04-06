df = read.csv("../data/grid_search_v1_10k/notes_and_data.csv")
colnames(df)
head(df)

acrobot.model= lm(X75 ~ factor(Has.target.network) + Initial.learning.rate + factor(learning.rate.annealing) + factor(Loss.Function) + weight.decay, df[df$Game=='Acrobot-v1',])

mountaincar.model= lm(X75 ~ factor(Has.target.network) + Initial.learning.rate + factor(learning.rate.annealing) + factor(Loss.Function) + weight.decay, df[df$Game=='MountainCar-v0',])

cartpole.model= lm(X75 ~ factor(Has.target.network) + Initial.learning.rate + factor(learning.rate.annealing) + factor(Loss.Function) + weight.decay, df[df$Game=='CartPole-v0',])

cartpole.model$coefficients[cartpole.model$coefficients > 0]

## no Target network, higher learning rate, turn on annealing, no MSE, higher weight decay

df1 = read.csv("../data/grid_search_v2_50k/results_new/notes_and_data.csv")

df2 = na.omit(read.csv("../data/grid_search_v2_50k/results_buggy/notes_and_data.csv"))

df = rbind(df1, df2)

acrobot.model= lm(mean ~ factor(model) + learning_rate + factor(weight_decay), df[df$game=='Acrobot-v1',])
summary(acrobot.model)
acrobot.model$coefficients[acrobot.model$coefficients < 0]

## acrobot likes DDQN***, lower lr, lower weight decay***

mountaincar.model= lm(mean ~ factor(model) + learning_rate + factor(weight_decay), df[df$game=='MountainCar-v0',])
summary(mountaincar.model)
mountaincar.model$coefficients[mountaincar.model$coefficients < 0]
## mountaincar likes DQN, lower lr, lower weight

cartpole.model= lm(mean ~ factor(model) + learning_rate+ factor(weight_decay), df[df$game=='CartPole-v0',])
summary(cartpole.model)
cartpole.model$coefficients[cartpole.model$coefficients > 0]
## cartpole likes DDQN***, higher lr***, lower weight decay**

## recommend: 0.1 weight decay, DDQN, different learning rates
