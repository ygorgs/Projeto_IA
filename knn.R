#Bibliotecas usadas
require(ggplot2)
require(caret)
require(dplyr)

temas <- read.csv2("data/temas.csv", encoding="UTF-8")

votacao <- read.csv2("data/deputados_temas_e_impeachment.csv", encoding="UTF-8")

votacao <- read.csv2("data/deputados_temas_e_impeachment_v1.1.csv", encoding="UTF-8")

#Tratamento dos dados (tirando os NA's) abstinencia a ausencias
votacao <- filter(votacao, grepl('SIM|NAO', votacao$IMPEACHMENT)) %>% droplevels()
votacao <- votacao[complete.cases(votacao),]
votacao$IMPEACHMENT <- as.factor(votacao$IMPEACHMENT)

#criando partições
split <- createDataPartition(y = votacao$IMPEACHMENT, p = 0.75, list = F)
treino <- votacao[split,]
teste <- votacao[-split,]

#adicionando cabeçalho
names(treino) = names(votacao)
names(teste) = names(votacao)

prop.table(table(treino$IMPEACHMENT))
prop.table(table(teste$IMPEACHMENT))


#treinando modelo
ctrl_knn <- trainControl(method = "repeatedcv", number = 10)
knnFit <- train(IMPEACHMENT ~ .-id_dep-nome-deputado-UF, 
                data = treino, 
                method="knn", 
                trControl = ctrl_knn, 
                preProcess = c("center","scale"), 
                tuneGrid = expand.grid(.k = 2:10),
                metric = "Accuracy")

#sumario do modelo
knnFit

#Realizando as predições
predicoes_knn <- predict(knnFit, teste)

confusionMatrix(teste$IMPEACHMENT, predicoes_knn)
