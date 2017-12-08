# roda modelos - paper Medeiros (2017)

# limpa workspace
rm(list=ls())

start_time = Sys.time()

# importa base de dados, funções e library
library(glmnet)
library(xlsx)
library(randomForest)

set.seed(101)

load("F:/COMUM/Gustavo/Aranha/data/BRinf.rda")
load("F:/COMUM/Gustavo/Aranha/data/consenso.rda")

source("F:/COMUM/Gustavo/Aranha/R/ic.glmnet.R")
source("F:/COMUM/Gustavo/Aranha/R/predict.ic.glmnet.R")
source("F:/COMUM/Gustavo/Aranha/R/csr.R")
source("F:/COMUM/Gustavo/Aranha/R/predict.csr.R")
source("F:/COMUM/Gustavo/Aranha/R/bagging.R")
source("F:/COMUM/Gustavo/Aranha/R/baggit.R")
source("F:/COMUM/Gustavo/Aranha/R/predict.bagging.R")

# prepara os dados
aux = embed(BRinf,2)
y = aux[,1]
x = aux[,-1]

# parâmetros
n = dim(x)[1]
m = dim(x)[2]
janela = 5*12
fim = n - 1
models = c('Ridge', 'Lasso', 'adaLasso', 'CSR', 'Random Forest')
qtd_h = 12

# matrizes
pred = array(NA, c(n, length(models), qtd_h))
ea = array(NA, c(n, length(models), qtd_h))

colnames(pred) = models
colnames(ea) = models


# loop backtest
for (i in 1:qtd_h) {
  for (j in (janela+1):(fim-i+1)) {
    y.in = y[(j-janela-1+i-1):(j-i)]; y.out=y[j]
    x.in = x[(j-janela-1+i-1):(j-i),]; x.out=x[j,]
    
    # ridge
    model_ridge = ic.glmnet(x.in,y.in,crit = "bic",alpha=0)
    pred[j, 'Ridge', i] = predict(model_ridge,newdata=x.out)
    ea[j, 'Ridge', i]   = abs(y[j] - pred[j, 'Ridge', i]) * 1000

    # lasso
    model_lasso = ic.glmnet(x.in,y.in,crit = "bic")
    pred[j, 'Lasso', i] = predict(model_lasso,newdata=x.out)
    ea[j, 'Lasso', i]         = abs(y[j] - pred[j, 'Lasso', i]) * 1000

    # # ada-lasso
    tau=1
    first.step.coef=coef(model_lasso)[-1]
    penalty.factor=abs(first.step.coef+1/sqrt(nrow(x)))^(-tau)
    model_adalasso=ic.glmnet(x.in,y.in,crit="bic",penalty.factor=penalty.factor)
    pred[j, 'adaLasso', i] = predict(model_adalasso,newdata=x.out)
    ea[j, 'adaLasso', i]      = abs(y[j] - pred[j, 'adaLasso', i]) * 1000

    # csr
    model_csr = csr(x.in,y.in,K=20,k=4,fixed.controls = 1)
    pred[j, 'CSR', i] = predict(model_csr,newdata=x.out)
    ea[j, 'CSR', i]           = abs(y[j] - pred[j, 'CSR', i]) * 1000

    # # random forest
    model_rf = randomForest(x=x.in, y=y.in)
    pred[j, 'Random Forest', i] = predict(model_rf,newdata=x.out)
    ea[j, 'Random Forest', i] = abs(y[j] - pred[j, 'Random Forest', i]) * 1000

  }
}

end_time = Sys.time()
time_dif = end_time - start_time

# erro absoluto médio
eam = colMeans(ea, na.rm = TRUE)

# gráficos
barplot(eam, ylab = "Erro Absoluto Médio", xlab = "Modelo")
boxplot(ea[,,1])
