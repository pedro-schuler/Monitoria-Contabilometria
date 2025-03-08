import numpy as np
import pandas as pd
import statsmodels.api as sm

# Define se estamos utilizando um modelo log-log ou log-lin
loglog = False
loglin = True

# Arredondamento
# Define a precisão do arredondamento ao calcular B0 e B1
precision = 3
# Define a precisão do arredondamento ao calcular logaritmos
lnprecision = 3

# nome das variáveis
VariavelDependente = "Preço da Ação"
VariavelIndependente = "Gasto dos Pares"

# Cria uma tabela com os dados
dados_brutos = {
    VariavelDependente: [21.0, 23.5, 14.6, 13.1, 20.0, 18.1, 20.2, 18.5],
    VariavelIndependente: [35, 28, 60, 71, 44, 45, 58, 44],
}

if loglog:
    dados_brutos[VariavelIndependente] = np.round(
        np.log(dados_brutos[VariavelIndependente]), lnprecision
    )
    dados_brutos[VariavelDependente] = np.round(
        np.log(dados_brutos[VariavelDependente]), lnprecision
    )

if loglin:
    dados_brutos[VariavelDependente] = np.round(
        np.log(dados_brutos[VariavelDependente]), lnprecision
    )

# Cria um DataFrame com os dados
dadosDataFrame = pd.DataFrame(data=dados_brutos)
multiplyXY = dadosDataFrame[VariavelDependente].multiply(
    dadosDataFrame[VariavelIndependente]
)
xSquared = dadosDataFrame[VariavelIndependente].pow(2)
dadosDataFrame["X*Y"] = multiplyXY
dadosDataFrame["X^2"] = xSquared

# Adiciona uma linha com a soma de todas as colunas numéricas
print(dadosDataFrame)
print("-------------\n")

# Obtém os somatórios
somatorios = dadosDataFrame.select_dtypes(np.number).sum()
print("Somatórios")
print(somatorios)
print("-------------\n")

# Obtém o número de elementos
number_of_elements = len(dadosDataFrame.index)
print("Número de elementos =")
print(number_of_elements)
print("-------------\n")

# Calcula B1
B1 = np.round(
    (
        number_of_elements * dadosDataFrame["X*Y"].sum()
        - dadosDataFrame[VariavelIndependente].sum()
        * dadosDataFrame[VariavelDependente].sum()
    )
    / (
        number_of_elements * dadosDataFrame["X^2"].sum()
        - dadosDataFrame[VariavelIndependente].sum() ** 2
    ),
    precision,
)

print("B1 =")
print(B1)
print("-------------\n")

# Calcula B0
B0 = np.round(
    (dadosDataFrame[VariavelDependente].sum() / number_of_elements)
    - B1 * (dadosDataFrame[VariavelIndependente].sum() / number_of_elements),
    precision,
)

print("B0 =")
print(B0)
print("-------------\n")

# Utilizando o statsmodels
dados = pd.DataFrame(data=dados_brutos)

# Defina as Variáveis
x = dados[VariavelIndependente]
y = dados[VariavelDependente]

# Adicione o termo constante B0
x = sm.add_constant(x)

# Executa o modelo e realiza a função de ajuste (fit)
result = sm.OLS(y, x).fit()

# Sumário de resultados
print(result.summary())
