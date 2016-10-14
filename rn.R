#Bibliotecas usadas
require(ggplot2)
require(caret)
require(dplyr)
require(kohonen)

temas <- read.csv2("data/temas.csv", encoding="UTF-8")

votacao <- read.csv2("data/deputados_temas_e_impeachment_v1.1.csv", encoding="UTF-8")

#Tratamento dos dados (tirando os NA's) abstinencia a ausencias
votacao <- filter(votacao, grepl('1|0', votacao$IMPEACHMENT)) %>% droplevels()
votacao <- votacao[complete.cases(votacao),]

#criando partições
split <- createDataPartition(y = votacao$IMPEACHMENT, p = 0.75, list = F)
treino <- votacao[split,]
treino <- treino[,c("tema_1","tema_2","tema_3","tema_4","tema_5","tema_6","tema_7","tema_8","tema_9",
                    "tema_10","tema_11","tema_12","tema_13","tema_14","tema_15","tema_16","tema_17",
                    "tema_18","tema_19","IMPEACHMENT")]


teste <- votacao[-split,]
teste <- teste[,c("tema_1","tema_2","tema_3","tema_4","tema_5","tema_6","tema_7","tema_8","tema_9",
                  "tema_10","tema_11","tema_12","tema_13","tema_14","tema_15","tema_16","tema_17",
                  "tema_18","tema_19","IMPEACHMENT")]

set.seed(7)

matrix.treino <- as.matrix(treino)
matrix.teste <- as.matrix(teste)

som.votacao <- som(matrix.treino[], grid=somgrid(5,5,"hexagonal"))

som.prediction <- predict(som.votacao, newdata=matrix.teste,
                          trainX = matrix.treino,
                          trainY = factor(votacao$IMPEACHMENT[split]))
confusionMatrix(factor(votacao$IMPEACHMENT[-split]), som.prediction$prediction)
