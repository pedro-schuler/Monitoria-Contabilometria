import numpy as np
import pandas as pd
import statsmodels.api as sm

# Define se estamos utilizando um modelo log-log ou log-lin
loglog = False
loglin = True

# Arredondamento
# Define a precisão do arredondamento ao calcular B0 e B1
precision = 5
# Define a precisão do arredondamento ao calcular logaritmos
lnprecision = 3

# Teste de Hipótese
hipoteseNula = 0

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
print("Dados iniciais:")
print(dadosDataFrame)
print("-------------\n")

# Adiciona colunas para X*Y e X^2
multiplyXY = dadosDataFrame[VariavelDependente].multiply(
    dadosDataFrame[VariavelIndependente]
)
xSquared = dadosDataFrame[VariavelIndependente].pow(2)
dadosDataFrame["X*Y"] = multiplyXY
dadosDataFrame["X^2"] = xSquared

# Exibe os dados
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

# Exibe o modelo calculado
print("Modelo Calculado:")
print(f"{VariavelDependente} = {B0} + {B1} * {VariavelIndependente}\n")
print("-------------\n")

# Calcula todos os Y's
dadosDataFrame["Y Calculado"] = np.round(
    dadosDataFrame[VariavelIndependente].apply(lambda x: x * B1 + B0), precision
)

# Calcula o quadrado resíduos
dadosDataFrame["U^2"] = np.round(
    (dadosDataFrame[VariavelDependente] - dadosDataFrame["Y Calculado"]) ** 2, precision
)

# Calcula o quadrado total
dadosDataFrame["(Y - Ybarra)^2"] = np.round(
    (dadosDataFrame[VariavelDependente] - dadosDataFrame[VariavelDependente].mean())
    ** 2,
    precision,
)

# Exibe os dados
print(dadosDataFrame)
print("-------------\n")

# Calcula o coeficiente de determinação
print("Coeficiente de Determinação =")
SQR = np.round(dadosDataFrame["U^2"].sum(), precision)
SQT = np.round(dadosDataFrame["(Y - Ybarra)^2"].sum(), precision)
R2 = np.round(1 - SQR / SQT, precision)
print(f"SQR = {SQR}")
print(f"SQT = {SQT}")
print(f"R2 = {R2}")
print("-------------\n")

# Calcula o R^2 ajustado
R2ajustado = np.round(
    1 - (SQR / (number_of_elements - 1)) / (SQT / (number_of_elements - 1)), precision
)
print(f"R^2 Ajustado = {R2ajustado}")
print("-------------\n")

# Teste de hipótese
S2 = SQR / (number_of_elements - 2)
dadosDataFrame["somatorioQuadradoX"] = np.round(
    (dadosDataFrame[VariavelIndependente] - dadosDataFrame[VariavelIndependente].mean())
    ** 2,
    precision,
)

# Calcula o teste T
print("Teste T(calculado) =")
TC = np.round(
    (B1 - hipoteseNula) / np.sqrt(S2 / dadosDataFrame["somatorioQuadradoX"].sum()),
    precision,
)
TCabsolute = np.abs(TC)
print(f"Tc = {TC}")
print(f"|Tc| = {TCabsolute}")
print("-------------\n")

# Calcula o Teste F
print("Teste F(calculado) =")
FC = np.round(TC**2, precision)
FCabsolute = np.abs(FC)
print(f"Fc = Tc^2 = {FC}")
print(f"|Fc| = |Tc^2| = {FCabsolute}")
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
print("-------------\n")

# Realiza o teste T
print("Teste T(statsmodels) =")
print(result.t_test([0, 1]))
print("-------------\n")

## Realiza o teste F
print("Teste F(statsmodels) =")
print(result.f_test([0, 1]))
