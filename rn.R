set.seed(500)

library(neuralnet)
library(ggplot2)
library(dplyr)

#le os dados
data <- read.csv2("data/deputados_temas_e_impeachment_v1.1.csv", encoding="UTF-8")

#Checa se há algum dado faltando
apply(data,2,function(x) sum(is.na(x)))

#Retira todos os NA
data <- data[complete.cases(data),]
data <- data[,c("tema_1","tema_2","tema_3","tema_4","tema_5","tema_6","tema_7","tema_8","tema_9",
                "tema_10","tema_11","tema_12","tema_13","tema_14","tema_15","tema_16","tema_17",
                "tema_18","tema_19","IMPEACHMENT")]
data <- filter(data, grepl('1|0', data$IMPEACHMENT)) %>% droplevels()

#KNN
#data$IMPEACHMENT <- as.factor(data$IMPEACHMENT)
#index <- sample(1:nrow(data),round(0.75*nrow(data)))
#train <- data[index,]
#test <- data[-index,]
#ctrl_knn <- trainControl(method = "repeatedcv", number = 10)
#knnFit <- train(IMPEACHMENT ~ ., 
#                data = train, 
#                method="knn", 
#                trControl = ctrl_knn, 
#                preProcess = c("center","scale"), 
#                tuneGrid = expand.grid(.k = 2:10),
#                metric = "Accuracy")

#pr.knn <- predict(knnFit,test)
#MSE.knn <- sum((pr.knn - test$IMPEACHMENT)^2)/nrow(test)



#Escalando os dados
maxs <- apply(data, 2, max)
mins <- apply(data, 2, min)
scaled <- as.data.frame(scale(data, center = mins, scale = maxs - mins))

#Separando as partições de treino e teste
index <- sample(1:nrow(data),round(0.75*nrow(data)))
train_ <- scaled[index,]
test_ <- scaled[-index,]

#Criando rede neural
n <- names(train_)
f <- as.formula(paste("IMPEACHMENT ~", paste(n[!n %in% "IMPEACHMENT"], collapse = " + ")))
nn <- neuralnet(f,data=train_,hidden=c(5,3),linear.output=T)

#representação gráfica do modelo com os pesos em cada conexão
plot(nn)

pr.nn <- compute(nn,test_[,1:19])
pr.nn_ <- pr.nn$net.result*(max(data$IMPEACHMENT)-min(data$IMPEACHMENT))+min(data$IMPEACHMENT)

#arredondando os valores
pr.nn_[pr.nn_ >=0.5] <- 1
pr.nn_[pr.nn_ <0.5] <- 0

test.r <- (test_$IMPEACHMENT)*(max(data$IMPEACHMENT)-min(data$IMPEACHMENT))+min(data$IMPEACHMENT)
MSE.nn <- sum((test.r - pr.nn_)^2)/nrow(test_)

par(mfrow=c(1,2))

plot(test$IMPEACHMENT,pr.nn_,col='red',main='Real vs predicted NN',pch=18,cex=0.7)
abline(0,1,lwd=2)
legend('bottomright',legend='NN',pch=18,col='red', bty='n')

#plot(test$IMPEACHMENT,pr.knn,col='blue',main='Real vs predicted lm',pch=18, cex=0.7)
#abline(0,1,lwd=2)
#legend('bottomright',legend='LM',pch=18,col='blue', bty='n', cex=.95)

confusionMatrix(test_$IMPEACHMENT, pr.nn_)
