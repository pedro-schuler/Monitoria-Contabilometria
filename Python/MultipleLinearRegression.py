import numpy as np
import pandas as pd
import statsmodels.api as sm

# Define se estamos utilizando um modelo log-log ou log-lin
loglog = False
loglin = True

# Define os nomes da variáveis
VDependente = "Preço da Ação"
VIndependente1 = "Gastos dos Pares"
VIndependente2 = "Habilidade do CFO"

# Cria uma tabela com os dados
dados_brutos = {
    VDependente: [21.0, 23.5, 14.6, 13.1, 20.0, 18.1, 20.2, 18.5],
    VIndependente1: [35, 28, 60, 71, 44, 45, 58, 44],
    VIndependente2: [5.0, 6.2, 9.5, 8.0, 7.9, 7.0, 8.5, 6.8],
}

dadosDataFrame = pd.DataFrame(data=dados_brutos)

# Visualiza os dados
print("Tabela com os dados")
print(dadosDataFrame)
print("-------\n")

# Se utilizarmos um modelo log-lin, a variável dependente será log-linearizada
if loglin:
    dados_brutos[VDependente] = np.log(dados_brutos[VDependente])

# Se utilizarmos um modelo log-log, todas as variáveis serão log-linearizadas
if loglog:
    dados_brutos[VDependente] = np.log(dados_brutos[VDependente])
    dados_brutos[VIndependente1] = np.log(dados_brutos[VIndependente1])
    dados_brutos[VIndependente2] = np.log(dados_brutos[VIndependente2])

## Fazendo todos os cálculos de maneira manual
# Defina a matriz Inicial
PrimeiroElemento = np.full(len(dados_brutos[VDependente]), 1)
VariavelIndependenteTransposta = np.array(
    [PrimeiroElemento, dados_brutos[VIndependente1], dados_brutos[VIndependente2]]
)
VariavelIndependente = np.transpose(VariavelIndependenteTransposta)
VariavelDependente = dados_brutos[VDependente]

print("Matriz das variáveis independentes\nX =")
print(VariavelIndependente)
print("-------\n")
print("Matriz transposta das variáveis independentes\nX' =")
print(VariavelIndependenteTransposta)
print("-------\n")

# Multiplica as matrizes
MultiplicacaoVariavelIndependente = np.matmul(
    VariavelIndependenteTransposta, VariavelIndependente
)
print(
    "Produto da matriz transposta das variáveis independentes pela matriz das variáveis independentes\nX'X ="
)
print(MultiplicacaoVariavelIndependente)
print("-------\n")

# Inverte a multiplicação das matrizes
MultiplicacaoVariavelIndependenteInversa = np.linalg.inv(
    MultiplicacaoVariavelIndependente
)
print("Matriz inversa do produto anterior\n(X'X)^-1 =")
print(MultiplicacaoVariavelIndependenteInversa)
print("-------\n")

# Multiplica pelo Y
MultiplicacaoVariavelIndependenteDependente = np.matmul(
    VariavelIndependenteTransposta, VariavelDependente
)
print(
    "Produto da matriz transposta das variáveis independentes pela matriz da variável dependente\nX'Y ="
)
print(MultiplicacaoVariavelIndependenteDependente)
print("-------\n")

# Obtém o resultado
Resultado = np.matmul(
    MultiplicacaoVariavelIndependenteInversa,
    MultiplicacaoVariavelIndependenteDependente,
)
print("Matriz contendo todos os Betas\nB = (X'X)^-1 * X'Y =")
print(Resultado)
print("-------\n")

## Utilizando o Statsmodels para fazer todos os cálculos
# Cria um dataframe com os dados
dados = pd.DataFrame(data=dados_brutos)

# Defina as Variáveis
x = np.array(dados[[VIndependente1, VIndependente2]])
y = dados[VDependente]

# Adicione o termo constante B0
x = sm.add_constant(x)

# Executa o modelo e realiza a função de ajuste (fit)
result = sm.OLS(y, x).fit()

# Obtém a matriz de variância Covariância
VarianceCovariance = result.cov_params()

# Matriz de variância Covariância
print("Matriz de variância Covariância")
print(VarianceCovariance)
print("-------\n")

# Sumário de resultados
print(result.summary())

# Executa o teste T
print("Teste T =")
print(result.t_test([0, 1, 0]))
print("-------\n")

# Executa o teste F
print("Teste F =")
print(result.f_test(np.identity(3)))
