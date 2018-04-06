df = read.csv("notes_and_data.csv")
colnames(df)
head(df)

acrobot.model= lm(X75 ~ factor(Has.target.network) + Initial.learning.rate + factor(learning.rate.annealing) + factor(Loss.Function) + weight.decay, df[df$Game=='Acrobot-v1',])

mountaincar.model= lm(X75 ~ factor(Has.target.network) + Initial.learning.rate + factor(learning.rate.annealing) + factor(Loss.Function) + weight.decay, df[df$Game=='MountainCar-v0',])

cartpole.model= lm(X75 ~ factor(Has.target.network) + Initial.learning.rate + factor(learning.rate.annealing) + factor(Loss.Function) + weight.decay, df[df$Game=='CartPole-v0',])

cartpole.model$coefficients

