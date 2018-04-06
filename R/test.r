df = read.csv("../data/grid_search_v1_10k/notes_and_data.csv")
colnames(df)
head(df)

acrobot.model= lm(X75 ~ factor(Has.target.network) + Initial.learning.rate + factor(learning.rate.annealing) + factor(Loss.Function) + weight.decay, df[df$Game=='Acrobot-v1',])

mountaincar.model= lm(X75 ~ factor(Has.target.network) + Initial.learning.rate + factor(learning.rate.annealing) + factor(Loss.Function) + weight.decay, df[df$Game=='MountainCar-v0',])

cartpole.model= lm(X75 ~ factor(Has.target.network) + Initial.learning.rate + factor(learning.rate.annealing) + factor(Loss.Function) + weight.decay, df[df$Game=='CartPole-v0',])

cartpole.model$coefficients[cartpole.model$coefficients > 0]

df = read.csv("../data/grid_search_v2_50k/results_new/notes_and_data.csv")
colnames(df)
head(df)

acrobot.model= lm(mean ~ factor(model) + learning_rate + weight_decay, df[df$game=='Acrobot-v1',])
summary(acrobot.model)

mountaincar.model= lm(mean ~ factor(model) + learning_rate + weight_decay, df[df$game=='MountainCar-v0',])
summary(mountaincar.model)

cartpole.model= lm(mean ~ factor(model) + learning_rate+ weight_decay, df[df$game=='CartPole-v0',])
summary(cartpole.model)

