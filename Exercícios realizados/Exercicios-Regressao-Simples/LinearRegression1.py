import pandas as pd
import statsmodels.api as sm

# Cria uma tabela com os dados
dados_brutos = {
    "X": [
        5.5,
        4.8,
        4.7,
        3.9,
        4.5,
        6.2,
        6.0,
        5.2,
        4.7,
        4.3,
        4.9,
        5.4,
        5.0,
        6.3,
        4.6,
        4.3,
        5.0,
        5.9,
        4.1,
        4.7,
    ],
    "Y": [
        3.1,
        2.3,
        3.0,
        1.9,
        2.5,
        3.7,
        3.4,
        2.6,
        2.8,
        1.6,
        2.0,
        2.9,
        2.3,
        3.2,
        1.8,
        1.4,
        2.0,
        3.8,
        2.2,
        1.5,
    ],
}
dados = pd.DataFrame(data=dados_brutos)

# Defina as Variáveis
x = dados["X"].tolist()
y = dados["Y"].tolist()

# Adicione o termo constante B0
x = sm.add_constant(x)

# Executa o modelo e realiza a função de ajuste (fit)
result = sm.OLS(y, x).fit()

# Sumário de resultados
print(result.summary())
