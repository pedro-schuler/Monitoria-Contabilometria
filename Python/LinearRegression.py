import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf


class LinearRegressionModel:
    def __init__(
        self,
        data,
        dependent_var,
        independent_var,
        loglog=False,
        loglin=False,
        precision=5,
        lnprecision=3,
        null_hypothesis=0,
    ):
        self.data = data
        self.dependent_var = dependent_var
        self.independent_var = independent_var
        self.loglog = loglog
        self.loglin = loglin
        self.precision = precision
        self.lnprecision = lnprecision
        self.null_hypothesis = null_hypothesis
        self.number_of_elements = len(data[dependent_var])
        self.transformed_data = None
        self.all_sums = None
        self.B0 = None
        self.B1 = None

    def initial_data(self):
        print("Dados iniciais:")
        print(pd.DataFrame(self.data))
        print("\n")
        print(f"Variável dependente: {self.dependent_var}")
        print(f"Variável independente: {self.independent_var}")
        print(f"Número de elementos: {self.number_of_elements}")
        print("-------------\n")

    def transform_data(self):
        print("Ajustando os dados...")
        data = self.data.copy()
        if self.loglog:
            data[self.independent_var] = np.round(
                np.log(data[self.independent_var]), self.lnprecision
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

    def calculate_products(self):
        print("Calculando os produtos X*Y e X^2...\n")
        df = self.transformed_data
        df["X*Y"] = df[self.dependent_var] * df[self.independent_var]
        df["X^2"] = df[self.independent_var] ** 2
        self.transformed_data = df
        print("Dados transformados:")
        print(self.transformed_data)
        print("-------------\n")

    def calculate_sums(self):
        print("Calculando somatórios...\n")
        df = self.transformed_data
        sum_xy = df["X*Y"].sum()
        sum_x = df[self.independent_var].sum()
        sum_y = df[self.dependent_var].sum()
        sum_x2 = df["X^2"].sum()

        all_sums = {
            "Soma Y": sum_y,
            "Soma X": sum_x,
            "Soma XY": sum_xy,
            "Soma X^2": sum_x2,
        }

        self.all_sums = pd.DataFrame(all_sums, index=[0])
        print(self.all_sums)
        print("-------------\n")

    def calculate_coefficients(self):
        print("Calculando os coeficientes B0 e B1...\n")
        sums = self.all_sums

        self.B1 = np.round(
            (
                self.number_of_elements * sums["Soma XY"].values[0]
                - sums["Soma X"].values[0] * sums["Soma Y"].values[0]
            )
            / (
                self.number_of_elements * sums["Soma X^2"].values[0]
                - sums["Soma X"].values[0] ** 2
            ),
            self.precision,
        )
        self.B0 = np.round(
            (sums["Soma Y"].values[0] / self.number_of_elements)
            - self.B1 * (sums["Soma X"].values[0] / self.number_of_elements),
            self.precision,
        )
        print(f"B1 = {self.B1}")
        print(f"B0 = {self.B0}")
        print("-------------\n")

    def show_model(self):
        print("O modelo obtido foi:\n")
        if self.loglog:
            print(f"ln(Y) = {self.B0} + {self.B1} * ln(X)")
        elif self.loglin:
            print(f"Y = {self.B0} + {self.B1} * ln(X)")
        else:
            print(f"Y = {self.B0} + {self.B1} * X")
        print("-------------\n")

    def calculate_residuals_and_r2(self):
        print("Calculando os resíduos e R²...\n")
        df = self.transformed_data
        df["Y Calculado"] = np.round(
            df[self.independent_var] * self.B1 + self.B0,
            self.precision,
        )
        df["U^2"] = np.round(
            (df[self.dependent_var] - df["Y Calculado"]) ** 2, self.precision
        )
        df["(Y - Ybarra)^2"] = np.round(
            (df[self.dependent_var] - df[self.dependent_var].mean()) ** 2,
            self.precision,
        )
        self.transformed_data = df
        print("Dados com resíduos:")
        print(self.transformed_data)
        print("-------------\n")

        print("Calculando SQR, SQT e R²...\n")
        SQR = df["U^2"].sum()
        SQT = df["(Y - Ybarra)^2"].sum()
        R2 = np.round(1 - SQR / SQT, self.precision)
        print(f"SQR = {SQR}")
        print(f"SQT = {SQT}")
        print(f"R² = {R2}")
        print("-------------\n")

    def hypothesis_tests(self):
        print("Executando os teste de hipótese...\n")
        df = self.transformed_data
        SQR = df["U^2"].sum()
        S2 = SQR / (self.number_of_elements - 2)
        somatorio_quadrado_x = np.round(
            (df[self.independent_var] - df[self.independent_var].mean()) ** 2,
            self.precision,
        )
        TC = np.round(
            (self.B1 - self.null_hypothesis) / np.sqrt(S2 / somatorio_quadrado_x.sum()),
            self.precision,
        )
        FC = np.round(TC**2, self.precision)

        print("Teste T:")
        print(f"Tc = {TC}")
        print(f"|Tc| = {np.abs(TC)}\n")
        print("Teste F:")
        print(f"Fc = {FC}")
        print(f"|Fc| = {np.abs(FC)}")
        print("-------------\n")

    def validate_with_statsmodels(self):
        print("Validando os dados usando o statsmodels...")

        formula = f"Q('{self.dependent_var}') ~ Q('{self.independent_var}')"
        hypothesis = f"Q('{self.independent_var}') = {self.null_hypothesis}"

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

    def plot_graph(self):
        print("Plotando o gráfico...\n")
        sns.set_theme()
        sns.lmplot(
            x=self.independent_var,
            y=self.dependent_var,
            data=self.transformed_data,
            ci=None,
            line_kws={"color": "red"},
        )
        plt.title("Gráfico de Regressão Linear")
        plt.xlabel(self.independent_var)
        plt.ylabel(self.dependent_var)
        plt.show()
        print("-------------\n")


# Example usage
data = {
    "Preço da Ação": [21.0, 23.5, 14.6, 13.1, 20.0, 18.1, 20.2, 18.5],
    "Gasto dos Pares": [35, 28, 60, 71, 44, 45, 58, 44],
}
model = LinearRegressionModel(data, "Preço da Ação", "Gasto dos Pares", loglin=True)
model.initial_data()
model.transform_data()
model.calculate_products()
model.calculate_sums()
model.calculate_coefficients()
model.show_model()
model.calculate_residuals_and_r2()
model.hypothesis_tests()
model.validate_with_statsmodels()
model.plot_graph()
