import pandas as pd
import statsmodels.api as sm

# Cria uma tabela com os dados
dados_brutos = {
    "GE": [5.27, 18, 0.59, 1.05, 2.52, 0.88],
    "GA": [1.03, 1.61, 0.7, 0.64, 1.01, 0.57],
}
dados = pd.DataFrame(data=dados_brutos)

# Defina as Variáveis
x = dados["GE"]
y = dados["GA"]

# Adicione o termo constante B0
x = sm.add_constant(x)

# Executa o modelo e realiza a função de ajuste (fit)
result = sm.OLS(y, x).fit()

# Sumário de resultados
print(result.summary())
