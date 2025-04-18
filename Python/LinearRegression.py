from typing import Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf


class LinearRegressionModel:
    """
    A class to perform linear regression analysis with various transformations and diagnostics.

    This model supports standard linear regression as well as log-log and log-linear transformations,
    and provides comprehensive statistical analysis including residuals, R², and hypothesis tests.
    """

    def __init__(
        self,
        data: Dict[str, List[float]] or pd.DataFrame,
        dependent_var: str,
        independent_var: str,
        loglog: bool = False,
        loglin: bool = False,
        precision: int = 5,
        lnprecision: int = 3,
        null_hypothesis: float = 0,
        verbose: bool = True,
    ):
        """
        Initialize the linear regression model.

        Parameters:
            data: Dictionary or DataFrame containing the data
            dependent_var: Name of the dependent variable (Y)
            independent_var: Name of the independent variable (X)
            loglog: Whether to apply log transformation to both X and Y
            loglin: Whether to apply log transformation to Y only
            precision: Decimal precision for calculations
            lnprecision: Decimal precision for log transformations
            null_hypothesis: Value for the null hypothesis (H₀: β₁ = value)
            verbose: Whether to print detailed information
        """
        # Convert data to DataFrame if it's a dictionary
        if isinstance(data, dict):
            self.data = pd.DataFrame(data)
        else:
            self.data = data.copy()

        self.dependent_var = dependent_var
        self.independent_var = independent_var
        self.loglog = loglog
        self.loglin = loglin
        self.precision = precision
        self.lnprecision = lnprecision
        self.null_hypothesis = null_hypothesis
        self.verbose = verbose

        # Validate data
        self._validate_inputs()

        self.number_of_elements = len(self.data[dependent_var])
        self.transformed_data = None
        self.all_sums = None
        self.B0 = None  # Intercept
        self.B1 = None  # Slope
        self.R2 = None  # R-squared
        self.stats_results = None  # For storing statsmodels results

    def _validate_inputs(self):
        """Validate input data and parameters."""
        # Check if variables exist in the dataset
        if self.dependent_var not in self.data.columns:
            raise ValueError(
                f"Variável dependente '{self.dependent_var}' não encontrada nos dados"
            )
        if self.independent_var not in self.data.columns:
            raise ValueError(
                f"Variável independente '{self.independent_var}' não encontrada nos dados"
            )

        # Check for non-positive values when using log transformations
        if self.loglog or self.loglin:
            if self.loglog and (self.data[self.independent_var] <= 0).any():
                raise ValueError(
                    f"Não é possível aplicar logaritmo a valores não positivos em '{self.independent_var}'"
                )
            if (self.data[self.dependent_var] <= 0).any():
                raise ValueError(
                    f"Não é possível aplicar logaritmo a valores não positivos em '{self.dependent_var}'"
                )

        # Check if there's enough data for regression
        if len(self.data) < 3:
            raise ValueError(
                "Pelo menos 3 pontos de dados são necessários para regressão com teste de hipótese"
            )

    def _print(self, message, separator=False):
        """
        Print information according to verbosity level and add separator if requested.

        Parameters:
            message: Message to be printed
            separator: Whether to print a separator line after the message
        """
        # Always print but adjust detail level based on verbosity
        if self.verbose:
            print(message)
        elif isinstance(message, pd.DataFrame):
            print(f"Dados: {message.shape[0]} linhas x {message.shape[1]} colunas")
        elif not message.startswith("---"):
            concise_msg = message.split("\n")[0] if "\n" in message else message
            print(concise_msg)

        if separator:
            if self.verbose:
                print("-------------\n")

    def initial_data(self):
        """Display the initial data and parameters."""
        self._print("Dados iniciais:")
        self._print(pd.DataFrame(self.data))
        self._print("")
        self._print(f"Variável dependente: {self.dependent_var}")
        self._print(f"Variável independente: {self.independent_var}")
        self._print(f"Número de elementos: {self.number_of_elements}", separator=True)
        return self

    def transform_data(self):
        """Transform data based on selected regression type (log-log, log-linear)."""
        self._print("Ajustando os dados...")
        data = self.data.copy()

        try:
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
            self._print(self.transformed_data, separator=True)

        except Exception as e:
            raise RuntimeError(f"Erro na transformação de dados: {str(e)}")

        return self

    def calculate_products(self):
        """Calculate X*Y and X² for regression calculation."""
        self._print("Calculando os produtos X*Y e X^2...\n")

        if self.transformed_data is None:
            self.transform_data()

        df = self.transformed_data
        df["X*Y"] = df[self.dependent_var] * df[self.independent_var]
        df["X^2"] = df[self.independent_var] ** 2
        self.transformed_data = df

        self._print("Dados transformados:")
        self._print(self.transformed_data, separator=True)
        return self

    def calculate_sums(self):
        """Calculate sums required for regression coefficients."""
        self._print("Calculando somatórios...\n")

        if "X*Y" not in self.transformed_data.columns:
            self.calculate_products()

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
        self._print(self.all_sums, separator=True)
        return self

    def calculate_coefficients(self):
        """Calculate regression coefficients B0 (intercept) and B1 (slope)."""
        self._print("Calculando os coeficientes B0 e B1...\n")

        if self.all_sums is None:
            self.calculate_sums()

        sums = self.all_sums
        n = self.number_of_elements

        # Check for matrix inversibility
        determinant = (n * sums["Soma X^2"].values[0]) - (sums["Soma X"].values[0] ** 2)

        if abs(determinant) < 1e-10:
            raise ValueError(
                "Matriz não é inversível. As variáveis independentes podem ser colineares."
            )

        self.B1 = np.round(
            (
                n * sums["Soma XY"].values[0]
                - sums["Soma X"].values[0] * sums["Soma Y"].values[0]
            )
            / determinant,
            self.precision,
        )
        self.B0 = np.round(
            (sums["Soma Y"].values[0] / n) - self.B1 * (sums["Soma X"].values[0] / n),
            self.precision,
        )

        self._print(f"B1 = {self.B1}")
        self._print(f"B0 = {self.B0}", separator=True)
        return self

    def show_model(self):
        """Display the resulting regression model equation."""
        self._print("O modelo obtido foi:")

        if self.B0 is None or self.B1 is None:
            self.calculate_coefficients()

        if self.loglog:
            self._print(f"ln(Y) = {self.B0} + {self.B1} * ln(X)", separator=True)
        elif self.loglin:
            self._print(f"ln(Y) = {self.B0} + {self.B1} * X", separator=True)
        else:
            self._print(f"Y = {self.B0} + {self.B1} * X", separator=True)
        return self

    def calculate_residuals_and_r2(self):
        """Calculate residuals, SQR, SQT, and R² for model goodness of fit."""
        self._print("Calculando os resíduos e R²...\n")

        if self.B0 is None or self.B1 is None:
            self.calculate_coefficients()

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

        self._print("Dados com resíduos:")
        self._print(self.transformed_data)

        SQR = df["U^2"].sum()
        SQT = df["(Y - Ybarra)^2"].sum()

        # Check if SQT is zero or close to zero
        if abs(SQT) < 1e-10:
            self._print(
                "AVISO: Soma total de quadrados (SQT) é próxima de zero, R² pode não ser confiável"
            )
            self.R2 = 0
        else:
            self.R2 = np.round(1 - SQR / SQT, self.precision)

        degrees_of_freedom = self.number_of_elements - 2
        S2 = np.round(SQR / (degrees_of_freedom), self.precision)

        self._print(f"\nSQR = {SQR}")
        self._print(f"SQT = {SQT}")
        self._print(f"R² = {self.R2}")
        self._print(f"S² = {S2}", separator=True)
        return self

    def hypothesis_tests(self):
        """Perform hypothesis tests (t-test and F-test) for the regression coefficient."""
        self._print("Executando os teste de hipótese...\n")

        if "U^2" not in self.transformed_data.columns:
            self.calculate_residuals_and_r2()

        df = self.transformed_data
        SQR = df["U^2"].sum()

        # Check if we have enough degrees of freedom
        if self.number_of_elements <= 2:
            raise ValueError("Não há dados suficientes para o teste de hipótese")

        S2 = SQR / (self.number_of_elements - 2)
        somatorio_quadrado_x = np.round(
            (df[self.independent_var] - df[self.independent_var].mean()) ** 2,
            self.precision,
        )

        # Check if sum of squared X deviations is zero
        sum_sq_x = somatorio_quadrado_x.sum()
        if abs(sum_sq_x) < 1e-10:
            raise ValueError(
                "Soma dos desvios quadrados de X é próxima de zero, não é possível calcular estatísticas de teste"
            )

        TC = np.round(
            (self.B1 - self.null_hypothesis) / np.sqrt(S2 / sum_sq_x),
            self.precision,
        )
        FC = np.round(TC**2, self.precision)

        self._print("Teste T:")
        self._print(f"Tc = {TC}")
        self._print(f"|Tc| = {np.abs(TC)}")
        self._print("\nTeste F:")
        self._print(f"Fc = {FC}")
        self._print(f"|Fc| = {np.abs(FC)}", separator=True)
        return self

    def validate_with_statsmodels(self):
        """Validate results using statsmodels package."""
        self._print("Validando os dados usando o statsmodels...")

        formula = f"Q('{self.dependent_var}') ~ Q('{self.independent_var}')"
        hypothesis = f"Q('{self.independent_var}') = {self.null_hypothesis}"

        try:
            model = smf.ols(formula, self.transformed_data)
            result = model.fit()
            self.stats_results = result

            if self.verbose:
                self._print(result.summary())
                self._print("\nTeste-t (statsmodels):")
                self._print(result.t_test(hypothesis))
                self._print("\nTeste-f (statsmodels):")
                self._print(result.f_test(hypothesis), separator=True)
            else:
                self._print(
                    f"Statsmodels: coef={result.params[1]:.{self.precision}f}, p-value={result.pvalues[1]:.{self.precision}f}, R²={result.rsquared:.{self.precision}f}",
                    separator=True,
                )

        except Exception as e:
            self._print(f"Erro na validação com statsmodels: {str(e)}", separator=True)

        return self

    def plot_graph(self):
        """Plot the regression line and scatter plot of the data."""
        self._print("Plotando o gráfico...")

        try:
            sns.set_theme(style="whitegrid", palette="viridis", font="DejaVu Sans")
            sns.lmplot(
                x=self.independent_var,
                y=self.dependent_var,
                data=self.transformed_data,
                ci=None,
                line_kws={"color": "red"},
            )

            # Add model equation to the plot
            model_text = f"y = {self.B0} + {self.B1}x"
            if self.loglog:
                model_text = f"ln(y) =F-test {self.B0} + {self.B1}ln(x)"
            elif self.loglin:
                model_text = f"ln(y) = {self.B0} + {self.B1}x"

            r2_text = f"R² = {self.R2}"
            plt.annotate(
                model_text + "\n" + r2_text,
                xy=(0.05, 0.95),
                xycoords="axes fraction",
                fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7),
            )

            plt.title("Gráfico de Regressão Linear")
            plt.xlabel(self.independent_var)
            plt.ylabel(self.dependent_var)
            plt.tight_layout()
            plt.show()

            self._print("Gráfico gerado com sucesso", separator=True)

        except Exception as e:
            self._print(f"Erro ao plotar gráfico: {str(e)}", separator=True)

        return self

    def run_all(self):
        """Run the complete regression analysis pipeline."""
        try:
            (
                self.initial_data()
                .transform_data()
                .calculate_products()
                .calculate_sums()
                .calculate_coefficients()
                .show_model()
                .calculate_residuals_and_r2()
                .hypothesis_tests()
                .validate_with_statsmodels()
                .plot_graph()
            )
            return self
        except Exception as e:
            print(f"Erro na análise de regressão: {str(e)}")
            return None

    def get_summary(self):
        """Return a summary of the regression results."""
        if self.B0 is None or self.B1 is None:
            return "Modelo ainda não foi ajustado. Execute calculate_coefficients() primeiro."

        model_type = "Linear"
        equation = f"Y = {self.B0} + {self.B1} * X"

        if self.loglog:
            model_type = "Log-Log"
            equation = f"ln(Y) = {self.B0} + {self.B1} * ln(X)"
        elif self.loglin:
            model_type = "Log-Linear"
            equation = f"ln(Y) = {self.B0} + {self.B1} * X"

        summary = (
            f"Tipo de Regressão: {model_type}\n"
            f"Equação: {equation}\n"
            f"R²: {self.R2}\n"
            f"Número de observações: {self.number_of_elements}\n"
        )

        return summary


# Exemplo de uso
if __name__ == "__main__":
    data = {
        "Preço da Ação": [21.0, 23.5, 14.6, 13.1, 20.0, 18.1, 20.2, 18.5],
        "Gasto dos Pares": [35, 28, 60, 71, 44, 45, 58, 44],
    }
    # Exemplo de uso com saída detalhada
    model = LinearRegressionModel(
        data, "Preço da Ação", "Gasto dos Pares", loglin=True, verbose=True
    )
    model.run_all()
