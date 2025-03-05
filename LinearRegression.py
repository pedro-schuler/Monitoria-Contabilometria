import numpy as np
import pandas as pd
import statsmodels.api as sm

# Define se estamos utilizando um modelo log-log ou log-lin
loglog = False
loglin = True

# Cria uma tabela com os dados
dados_brutos = {
    "VariavelDependente": [21.0, 23.5, 14.6, 13.1, 20.0, 18.1, 20.2, 18.5],
    "VariavelIndependente": [35, 28, 60, 71, 44, 45, 58, 44],
}

if loglog:
    dados_brutos["VariavelIndependente"] = np.log(dados_brutos["VariavelIndependente"])
    dados_brutos["VariavelDependente"] = np.log(dados_brutos["VariavelDependente"])

if loglin:
    dados_brutos["VariavelDependente"] = np.log(dados_brutos["VariavelDependente"])

# Obtém o número de elementos
number_of_elements = len(dados_brutos["VariavelIndependente"])
print("Número de elementos =")
print(number_of_elements)
print("-------------\n")

# Multiplica os dois elementos para obter X*Y
XY = np.sum(
    np.multiply(
        dados_brutos["VariavelDependente"], dados_brutos["VariavelIndependente"]
    )
)

print("X * Y =")
print(XY)
print("-------------\n")

# Obtém a soma do quadrado de X
Xsquared = np.sum(np.square(dados_brutos["VariavelIndependente"]))

print("X^2 =")
print(Xsquared)
print("-------------\n")

# Obtém o somatório de X
Xsum = np.sum(dados_brutos["VariavelIndependente"])

print("Somatório de X =")
print(Xsum)
print("-------------\n")

# Obtém o somatório de Y
Ysum = np.sum(dados_brutos["VariavelDependente"])

print("Somatório de Y =")
print(Ysum)
print("-------------\n")

# Calcula B1
B1 = (number_of_elements * XY - Xsum * Ysum) / (number_of_elements * Xsquared - Xsum**2)

print("B1 =")
print(B1)
print("-------------\n")

# Calcula B0
B0 = (Ysum / number_of_elements) - B1 * (Xsum / number_of_elements)

print("B0 =")
print(B0)
print("-------------\n")

# Utilizando o statsmodels
dados = pd.DataFrame(data=dados_brutos)

# Defina as Variáveis
x = dados["VariavelIndependente"]
y = dados["VariavelDependente"]

# Adicione o termo constante B0
x = sm.add_constant(x)

# Executa o modelo e realiza a função de ajuste (fit)
result = sm.OLS(y, x).fit()

# Sumário de resultados
print(result.summary())
