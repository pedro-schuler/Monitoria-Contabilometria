import pandas as pd
import statsmodels.api as sm

# Cria uma tabela com os dados
dados_brutos = {
    "X": [
        7.0,
        6.0,
        5.0,
        1.0,
        5.0,
        4.0,
        7.0,
        3.0,
        4.0,
        2.0,
        8.0,
        5.0,
        2.0,
        5.0,
        7.0,
        1.0,
        4.0,
        5.0,
    ],
    "Y": [
        97.0,
        86.0,
        78.0,
        10.0,
        75.0,
        62.0,
        101.0,
        39.0,
        53.0,
        33.0,
        118.0,
        65.0,
        25.0,
        71.0,
        105.0,
        17.0,
        49.0,
        68.0,
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
