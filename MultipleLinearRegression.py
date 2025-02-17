import numpy as np
import pandas as pd
import statsmodels.api as sm

# Cria uma tabela com os dados
dados_brutos = {
    "VDependente": [10, 15, 30, 16, 20, 12],
    "VIndependente1": [25, 30, 35, 28, 25, 25],
    "VIndependente2": [22, 20, 24, 16, 18, 24],
}
dados = pd.DataFrame(data=dados_brutos)

# Defina a matriz inicial
VariavelIndependenteTransposta = np.array(
    [[1, 1, 1, 1, 1, 1], dados["VIndependente1"], dados["VIndependente2"]]
)
VariavelIndependente = np.transpose(VariavelIndependenteTransposta)
VariavelDependente = dados["VDependente"]

print("Matriz das variáveis independentes\n")
print(VariavelIndependente)
print("-------\n")
print("Matriz transposta das variáveis independentes\n")
print(VariavelIndependenteTransposta)
print("-------\n")

# Multiplica as matrizes
MultiplicacaoVariavelIndependente = np.matmul(
    VariavelIndependenteTransposta, VariavelIndependente
)
print(
    "Produto da matriz transposta das variáveis independentes pela matriz das variáveis indepentes\n"
)
print(MultiplicacaoVariavelIndependente)
print("-------\n")

# Inverte a multiplicação das matrizes
MultiplicacaoVariavelIndependenteInversa = np.linalg.inv(
    MultiplicacaoVariavelIndependente
)
print("Matriz inversa do produto anterior\n")
print(MultiplicacaoVariavelIndependenteInversa)
print("-------\n")

# Multiplica pelo Y
MultiplicacaoVariavelIndependenteDependente = np.matmul(
    VariavelIndependenteTransposta, VariavelDependente
)
print(
    "Produto da matriz transposta das variáveis independentes pela matriz da variável dependente\n"
)
print(MultiplicacaoVariavelIndependenteDependente)
print("-------\n")

# Obtem o resultado
Resultado = np.matmul(
    MultiplicacaoVariavelIndependenteInversa,
    MultiplicacaoVariavelIndependenteDependente,
)
print("Matriz contendo todos os Betas")
print(Resultado)
print("-------\n")

# Defina as Variáveis
x = np.array(dados[["VIndependente1", "VIndependente2"]])
y = dados["VDependente"]

# Adicione o termo constante B0
x = sm.add_constant(x)

# Executa o modelo e realiza a função de ajuste (fit)
result = sm.OLS(y, x).fit()

# Sumário de resultados
print(result.summary())
