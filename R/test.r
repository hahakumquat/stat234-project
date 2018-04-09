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

df = read.csv("../data/grid_search_v3_100k/notes_and_data.csv")

acrobot.model= lm(mean ~ factor(batch_size) + factor(initial_learning_rate), df[df$game=='Acrobot-v1',])
summary(acrobot.model)
acrobot.model$coefficients[acrobot.model$coefficients < 0]

## acrobot likes DDQN***, lower lr, lower weight decay***
mountaincar.model= lm(mean ~ factor(batch_size) + factor(initial_learning_rate), df[df$game=='MountainCar-v0',])
summary(mountaincar.model)
mountaincar.model$coefficients[mountaincar.model$coefficients < 0]
## mountaincar likes DQN, lower lr, lower weight

CartPole.model= lm(mean ~ factor(batch_size) + factor(initial_learning_rate), df[df$game=='CartPole-v0',])
summary(CartPole.model)
CartPole.model$coefficients[CartPole.model$coefficients < 0]

df = read.csv("../data/model_selection_100k/notes_and_data.csv", stringsAsFactors = FALSE)

df$model[df$model == "DQN_GS"] <- "DQCNN_GS"
df$model[df$model == "DDQN_GS"] <- "DDQCNN_GS"
df$model[df$model == "DDQCN_PCA_mini"] <- "DDQCNN_PCA_Mini"

new.df = data.frame(model=df$model, game=df$game,
                Convolution=grepl("CNN", df$model),
                Double=grepl("DD", df$model),
                PCA=grepl("PCA", df$model),
                Mini=grepl("Mini", df$model),
                mean=df$mean)

cartpole.model = lm(mean ~ Convolution + Double + PCA + Mini, new.df[new.df$game == "CartPole-v0",])

summary(cartpole.model)

acrobot.model = lm(mean ~ Convolution + Double + PCA + Mini, new.df[new.df$game == "Acrobot-v1",])

summary(acrobot.model)

mountaincar.model = lm(mean ~ Convolution + Double + PCA + Mini, new.df[new.df$game == "MountainCar-v0",])

summary(mountaincar.model)
