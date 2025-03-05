import pandas as pd
import statsmodels.api as sm

# Cria uma tabela com os dados
dados_brutos = {
    "VariavelIndependente": [8, 5, 15, 8, 20],
    "VariavelDependente": [20, 40, 5, 25, 2],
}
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
