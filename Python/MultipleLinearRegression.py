import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf

np.set_printoptions(precision=6, formatter={"float": "{: 0.6f}".format}, suppress=True)


class LinearRegressionModel:
    def __init__(
        self,
        data,
        dependent_var,
        first_independent_var,
        second_independent_var,
        loglog=False,
        loglin=False,
        precision=10,
        lnprecision=100,
        null_hypothesis=0,
    ):
        self.data = data
        self.dependent_var = dependent_var
        self.first_independent_var = first_independent_var
        self.second_independent_var = second_independent_var
        self.loglog = loglog
        self.loglin = loglin
        self.precision = precision
        self.lnprecision = lnprecision
        self.null_hypothesis = null_hypothesis
        self.number_of_elements = len(data[dependent_var])
        self.independent_vars_matrix = None
        self.independent_vars_inverse_matrix = None
        self.transposed_independent_vars_matrix = None
        self.dependent_vars_matrix = None
        self.B = None
        self.s2 = None

    def initial_data(self):
        print("Dados iniciais:")
        print(pd.DataFrame(self.data))
        print("\n")
        print(f"Variável dependente: {self.dependent_var}")
        print(f"Primeira variável independente: {self.first_independent_var}")
        print(f"Segunda variável independente: {self.second_independent_var}")
        print(f"Número de elementos: {self.number_of_elements}")
        print("-------------\n")

    def transform_data(self):
        print("Ajustando os dados...")
        data = self.data.copy()
        if self.loglog:
            data[self.first_independent_var] = np.round(
                np.log(data[self.first_independent_var]), self.lnprecision
            )
            data[self.second_independent_var] = np.round(
                np.log(data[self.second_independent_var]), self.lnprecision
            )
            data[self.dependent_var] = np.round(
                np.log(data[self.dependent_var]), self.lnprecision
            )
        elif self.loglin:
            data[self.dependent_var] = np.round(
                np.log(data[self.dependent_var]), self.lnprecision
            )
        self.transformed_data = pd.DataFrame(data)
        print(self.transformed_data)
        print("-------------\n")

    def calculate_vars_matrices(self):
        print("Calculando as matrizes de variáveis...\n")
        self.independent_vars_matrix = self.transformed_data[
            [self.first_independent_var, self.second_independent_var]
        ]
        self.independent_vars_matrix.insert(0, "1", 1)
        self.dependent_vars_matrix = self.transformed_data[[self.dependent_var]]
        print("Matriz de variáveis independentes:")
        print(self.independent_vars_matrix)
        print("\nMatriz de variáveis dependentes:")
        print(self.dependent_vars_matrix)
        print("-------------\n")

    def transpose_DataFrame(self, msg, dataframe):
        print(f"{msg}\n")
        transposed_dataframe = dataframe.transpose()
        print(transposed_dataframe)
        print("-------------\n")
        return transposed_dataframe

    def dataframe_to_matrix_multiplication(self, msg, DataFrame_a, DataFrame_b):
        print(f"{msg}")
        matrix_a = DataFrame_a.to_numpy()
        matrix_b = DataFrame_b.to_numpy()
        result = np.round(np.matmul(matrix_a, matrix_b), self.precision)
        print("Resultado da multiplicação:")
        print(result)
        print("-------------\n")
        return result

    def calculate_coefficients(self):
        multiply_independent_vars = self.dataframe_to_matrix_multiplication(
            "Calculando a matriz X'X...",
            self.transposed_independent_vars_matrix,
            self.independent_vars_matrix,
        )

        print("Calculando a matriz inversa de X'X...")
        self.independent_vars_inverse_matrix = np.linalg.inv(multiply_independent_vars)
        print(self.independent_vars_inverse_matrix)
        print("-------------\n")

        multiply_dependent_vars = self.dataframe_to_matrix_multiplication(
            "Calculando a matriz X'Y...",
            self.transposed_independent_vars_matrix,
            self.dependent_vars_matrix,
        )

        beta = np.round(
            np.matmul(self.independent_vars_inverse_matrix, multiply_dependent_vars),
            self.precision,
        )

        self.B = pd.DataFrame(beta, columns=["betas"])
        print(self.B)

    def show_model(self):
        print("\nO modelo obtido foi:")
        if self.loglog:
            print(
                f"ln(Y) = {self.B.iloc[0].values[0]} + {self.B.iloc[1].values[0]} * ln(X1) + {self.B.iloc[2].values[0]} * ln(X2)"
            )
        elif self.loglin:
            print(
                f"ln(Y) = {self.B.iloc[0].values[0]} + {self.B.iloc[1].values[0]} * X1 + {self.B.iloc[2].values[0]} * X2"
            )
        else:
            print(
                f"Y = {self.B.iloc[0].values[0]} + {self.B.iloc[1].values[0]} * X1 + {self.B.iloc[2].values[0]} * X2"
            )
        print("-------------\n")

    def calculate_residuals_and_s2(self):
        print("Calculando os resíduos e S²...\n")
        df = self.transformed_data
        df["Y Calculado"] = np.round(
            df[self.second_independent_var] * self.B.iloc[2].values[0]
            + df[self.first_independent_var] * self.B.iloc[1].values[0]
            + self.B.iloc[0].values[0],
            self.precision,
        )
        df["U^2"] = np.round(
            (df[self.dependent_var] - df["Y Calculado"]) ** 2, self.precision
        )
        self.transformed_data = df
        print("Dados com resíduos:")
        print(self.transformed_data)
        print("-------------\n")

        print("Calculando S²...\n")
        SQR = df["U^2"].sum()
        self.s2 = SQR / (self.number_of_elements - 2)
        print(f"S² = {self.s2}")
        print("-------------\n")

    def calculate_variance_covariance_matrix(self):
        print("Calculando a matriz de variância-covariância...\n")

        variance_covariance_matrix = self.s2 * self.independent_vars_inverse_matrix
        print("Matriz de variância-covariância:")
        print(variance_covariance_matrix)
        print("-------------\n")

    def validate_with_statsmodels(self):
        print("Validando os dados usando o statsmodels...")

        formula = f"Q('{self.dependent_var}') ~ Q('{self.first_independent_var}') + Q('{self.second_independent_var}')"
        hypothesis = f"Q('{self.first_independent_var}') = {self.null_hypothesis}, Q('{self.second_independent_var}') = {self.null_hypothesis}"

        model = smf.ols(formula, self.transformed_data)
        result = model.fit()

        print(result.summary())
        print("-------------\n")
        print("T-test (statsmodels):")
        print(result.t_test(hypothesis))
        print("-------------\n")
        print("F-test (statsmodels):")
        print(result.f_test(hypothesis))
        print("-------------\n")

    # def plot_graph(self):
    #     print("Plotando o gráfico...\n")
    #     sns.set_theme()
    #     sns.lmplot(
    #         x=self.first_independent_var,
    #         y=self.dependent_var,
    #         data=self.transformed_data,
    #         ci=None,
    #         line_kws={"color": "red"},
    #     )
    #     plt.title("Gráfico de Regressão Linear")
    #     plt.xlabel(self.first_independent_var)
    #     plt.ylabel(self.dependent_var)
    #     plt.show()
    #     print("-------------\n")
    #


# Exemplo de uso
data = {
    "Preço da Ação": [21.0, 23.5, 14.6, 13.1, 20.0, 18.1, 20.2, 18.5],
    "Gasto dos Pares": [35, 28, 60, 71, 44, 45, 58, 44],
    "Habilidade do CFO": [5.0, 6.2, 9.5, 8.0, 7.9, 7.0, 8.5, 6.8],
}
model = LinearRegressionModel(
    data, "Preço da Ação", "Gasto dos Pares", "Habilidade do CFO", loglin=True
)
model.initial_data()
model.transform_data()
model.calculate_vars_matrices()
model.transposed_independent_vars_matrix = model.transpose_DataFrame(
    "Calculando a matriz de variáveis independentes transposta...",
    model.independent_vars_matrix,
)
model.calculate_coefficients()
model.show_model()
model.calculate_residuals_and_s2()
model.calculate_variance_covariance_matrix()
model.validate_with_statsmodels()
# model.plot_graph()
