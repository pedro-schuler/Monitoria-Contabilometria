from fractions import Fraction

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from numpy.linalg import LinAlgError

np.set_printoptions(precision=6, formatter={"float": "{: 0.6f}".format}, suppress=True)


class LinearRegressionModel:
    """Multiple Linear Regression model with matrix operations and customizable output verbosity."""

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
        verbose=True,
    ):
        """
        Initialize the regression model with data and transformation parameters.

        Parameters:
        -----------
        data : dict or pandas.DataFrame
            The dataset containing dependent and independent variables
        dependent_var : str
            Name of the dependent variable column
        first_independent_var : str
            Name of the first independent variable column
        second_independent_var : str
            Name of the second independent variable column
        loglog : bool, optional
            If True, applies logarithmic transformation to both dependent and independent variables
        loglin : bool, optional
            If True, applies logarithmic transformation only to dependent variable
        precision : int, optional
            Decimal precision for calculations
        lnprecision : int, optional
            Decimal precision for logarithmic transformations
        null_hypothesis : float, optional
            Value for the null hypothesis in statistical tests
        verbose : bool, optional
            If True, displays detailed output at each step; if False, shows condensed output
        """
        self.data = pd.DataFrame(data) if isinstance(data, dict) else data.copy()
        self.dependent_var = dependent_var
        self.first_independent_var = first_independent_var
        self.second_independent_var = second_independent_var
        self.loglog = loglog
        self.loglin = loglin
        self.precision = precision
        self.lnprecision = lnprecision
        self.null_hypothesis = null_hypothesis
        self.verbose = verbose
        self.number_of_elements = len(self.data[dependent_var])

        # Initialize result attributes
        self.transformed_data = None
        self.independent_vars_matrix = None
        self.independent_vars_inverse_matrix = None
        self.transposed_independent_vars_matrix = None
        self.dependent_vars_matrix = None
        self.B = None
        self.s2 = None
        self.result_model = None

        # Validate input data
        self._validate_data()

    def _validate_data(self):
        """Validate the input data for the regression model."""
        required_columns = [
            self.dependent_var,
            self.first_independent_var,
            self.second_independent_var,
        ]
        missing_columns = [
            col for col in required_columns if col not in self.data.columns
        ]

        if missing_columns:
            raise ValueError(
                f"Colunas faltantes necessárias: {', '.join(missing_columns)}"
            )

        # Check for non-positive values if log transformation will be applied
        if self.loglog or self.loglin:
            vars_to_check = [self.dependent_var]
            if self.loglog:
                vars_to_check.extend(
                    [self.first_independent_var, self.second_independent_var]
                )

            for var in vars_to_check:
                if (self.data[var] <= 0).any():
                    raise ValueError(
                        f"Não é possível aplicar o logaritmo às seguintes variáveis com valores negativos: {var}"
                    )

    def _print_section(self, title, content=None, separator=True):
        """Helper method to standardize print formatting based on verbosity level."""
        print(f"{title}")

        if content is not None:
            if self.verbose:
                # Full output mode - print complete content
                print(content)
            else:
                # Concise mode - print summary or first few rows
                if isinstance(content, pd.DataFrame):
                    if len(content) > 5:
                        # Print first 3 rows and dimensions indicator
                        print(f"Formato do DataFrame: {content.shape}")
                        print(content.head(3))
                        print("... [mostrando 3/{} linhas] ...".format(len(content)))
                    else:
                        # Small DataFrame - print all rows
                        print(content)
                else:
                    # For non-DataFrame objects, print as is
                    print(content)

        if separator:
            print(
                "--------------------------------------------------------------------\n"
            )

    def initial_data(self):
        """Display the initial dataset and model configuration."""
        self._print_section("Dados iniciais:", self.data, separator=False)
        print(f"Variável dependente: {self.dependent_var}")
        print(f"Primeira variável independente: {self.first_independent_var}")
        print(f"Segunda variável independente: {self.second_independent_var}")
        print(f"Número de elementos: {self.number_of_elements}")
        self._print_section("", None)

    def transform_data(self):
        """Apply logarithmic transformations to the data if required."""
        data = self.data.copy()

        # Apply appropriate transformations based on model type
        if self.loglog or self.loglin:
            # Transform dependent variable for both loglog and loglin
            data[self.dependent_var] = np.round(
                np.log(data[self.dependent_var]), self.lnprecision
            )

            # Transform independent variables only for loglog
            if self.loglog:
                data[self.first_independent_var] = np.round(
                    np.log(data[self.first_independent_var]), self.lnprecision
                )
                data[self.second_independent_var] = np.round(
                    np.log(data[self.second_independent_var]), self.lnprecision
                )

        self.transformed_data = data
        self._print_section("Ajustando os dados...", self.transformed_data)

    def calculate_vars_matrices(self):
        """Create matrices for dependent and independent variables."""
        self._print_section("Calculando as matrizes de variáveis...\n", separator=False)

        # Create X matrix with intercept and Y matrix
        self.independent_vars_matrix = self.transformed_data[
            [self.first_independent_var, self.second_independent_var]
        ].copy()
        self.independent_vars_matrix.insert(0, "Const", 1)  # Add intercept column
        self.dependent_vars_matrix = self.transformed_data[[self.dependent_var]]

        self._print_section(
            "Matriz de variáveis independentes:",
            self.independent_vars_matrix,
            separator=False,
        )
        self._print_section(
            "\nMatriz de variáveis dependentes:", self.dependent_vars_matrix
        )

    def matrix_operation(self, operation_type, matrix_a, matrix_b=None, message=""):
        """Unified method for matrix operations with proper formatting."""
        self._print_section(message, separator=False)

        result = None

        if operation_type == "transpose":
            result = matrix_a.transpose()
            if self.verbose:
                print(result)
            else:
                print("[Transposição da matriz concluída]")
                if isinstance(result, pd.DataFrame) and not result.empty:
                    print(f"Formato: {result.shape}")

        elif operation_type == "multiply":
            if matrix_b is not None:
                a_numpy = matrix_a.to_numpy()
                b_numpy = matrix_b.to_numpy()
                result = np.round(np.matmul(a_numpy, b_numpy), self.precision)
                print("Resultado da multiplicação:")
                if self.verbose:
                    print(result)
                else:
                    print(f"Formato da matriz: {result.shape}")
                    if result.size <= 9:  # Show small matrices entirely
                        print(result)
                    else:
                        print(
                            "[Multiplicação da matriz concluída - exibindo resultados parciais]"
                        )

        elif operation_type == "inverse":
            try:
                result = np.linalg.inv(matrix_a)

                # Display the inverse matrix in fraction form
                frac_matrix = np.array(
                    [
                        [Fraction(float(val)).limit_denominator() for val in row]
                        for row in result
                    ]
                )

                if self.verbose:
                    for row in frac_matrix:
                        print([str(frac) for frac in row])
                else:
                    print("[Matriz inversa calculada]")
                    print(f"Formato: {result.shape}")
                    if result.size <= 9:  # Show small matrices entirely
                        print(result)
            except LinAlgError:
                error_msg = "A matriz não é inversível. Verifique a colinearidade entre as variáveis."
                print(f"ERRO: {error_msg}")
                raise LinAlgError(error_msg)

        return result

    def calculate_coefficients(self):
        """Calculate regression coefficients using the matrix approach."""
        # Calculate X'X
        xtx = self.matrix_operation(
            "multiply",
            self.transposed_independent_vars_matrix,
            self.independent_vars_matrix,
            "\nCalculando a matriz X'X...",
        )

        # Calculate (X'X)^-1
        self.independent_vars_inverse_matrix = self.matrix_operation(
            "inverse", xtx, message="\nCalculando a matriz inversa de X'X..."
        )

        # Calculate X'Y
        xty = self.matrix_operation(
            "multiply",
            self.transposed_independent_vars_matrix,
            self.dependent_vars_matrix,
            "\nCalculando a matriz X'Y...",
        )

        # Calculate beta coefficients: β = (X'X)^-1 * X'Y
        beta = np.round(
            np.matmul(self.independent_vars_inverse_matrix, xty), self.precision
        )
        self.B = pd.DataFrame(beta, columns=["betas"])
        self._print_section("\nCoeficientes de regressão (betas):", self.B)

    def show_model(self):
        """Display the fitted regression equation."""
        self._print_section("\nO modelo obtido foi:", separator=False)

        # Extract coefficients
        b0 = self.B.iloc[0].values[0]
        b1 = self.B.iloc[1].values[0]
        b2 = self.B.iloc[2].values[0]

        # Display appropriate model form based on transformation type
        if self.loglog:
            print(f"ln(Y) = {b0} + {b1} * ln(X1) + {b2} * ln(X2)")
        elif self.loglin:
            print(f"ln(Y) = {b0} + {b1} * X1 + {b2} * X2")
        else:
            print(f"Y = {b0} + {b1} * X1 + {b2} * X2")

        self._print_section("", None)

    def calculate_residuals_and_s2(self):
        """Calculate residuals and the error variance (S²)."""
        self._print_section("Calculando os resíduos e S²...\n", separator=False)

        df = self.transformed_data.copy()
        b0, b1, b2 = [self.B.iloc[i].values[0] for i in range(3)]

        # Calculate fitted values
        df["Y Calculado"] = np.round(
            b0
            + b1 * df[self.first_independent_var]
            + b2 * df[self.second_independent_var],
            self.precision,
        )

        # Calculate squared residuals
        df["U^2"] = np.round(
            (df[self.dependent_var] - df["Y Calculado"]) ** 2, self.precision
        )
        self.transformed_data = df

        self._print_section(
            "Dados com resíduos...", self.transformed_data, separator=False
        )

        # Calculate S² (error variance)
        SQR = df["U^2"].sum()
        degrees_of_freedom = (
            self.number_of_elements - 3
        )  # n-k (k=3 including intercept)
        self.s2 = SQR / degrees_of_freedom
        self._print_section("\nCalculando S²", f"S² = {self.s2}")

    def calculate_variance_covariance_matrix(self):
        """Calculate the variance-covariance matrix for coefficient estimates."""
        self._print_section(
            "Calculando a matriz de variância-covariância...\n", separator=False
        )

        variance_covariance_matrix = self.s2 * self.independent_vars_inverse_matrix
        if self.verbose:
            self._print_section(
                "Matriz de variância-covariância", variance_covariance_matrix
            )
        else:
            if variance_covariance_matrix.size <= 9:  # Show small matrices entirely
                self._print_section(
                    "Matriz de variância-covariância", variance_covariance_matrix
                )
            else:
                print(f"Formato da matriz: {variance_covariance_matrix.shape}")
                print("[Mostrando um excerto da matriz de variância-covariância]")
                self._print_section("", None)

    def validate_with_statsmodels(self):
        """Validate the regression results using statsmodels package."""
        self._print_section(
            "Validando os dados usando o statsmodels...", separator=False
        )

        # Prepare model formula and hypothesis
        formula = f"Q('{self.dependent_var}') ~ Q('{self.first_independent_var}') + Q('{self.second_independent_var}')"
        hypothesis = f"Q('{self.first_independent_var}') = {self.null_hypothesis}, Q('{self.second_independent_var}') = {self.null_hypothesis}"

        # Fit the model
        try:
            model = smf.ols(formula, self.transformed_data)
            self.result_model = model.fit()

            # Display results
            if self.verbose:
                print(self.result_model.summary())
            else:
                # Show just key statistics in concise mode
                print("\nEstatísticas importantes do modelo:")
                print(f"R-quadrado: {self.result_model.rsquared:.4f}")
                print(f"R-quadrado ajustado: {self.result_model.rsquared_adj:.4f}")
                print(
                    f"F-estatística: {self.result_model.fvalue:.4f} (p-valor: {self.result_model.f_pvalue:.4f})"
                )
                print("\nCoeficientes:")
                coef_summary = self.result_model.summary2().tables[1][
                    ["Coef.", "P>|t|"]
                ]
                print(coef_summary)

            print("\n")
            print("Matriz de variância-covariância (statsmodels):")
            if self.verbose:
                print(self.result_model.cov_params())
            else:
                print(
                    "[Dimensões da matriz: {}]".format(
                        self.result_model.cov_params().shape
                    )
                )

            print("\n")
            print("Teste-t (statsmodels):")
            t_test_result = self.result_model.t_test(hypothesis)
            if self.verbose:
                print(t_test_result)
            else:
                print(f"t-estatística: {t_test_result.tvalue[0][0]:.4f}")
                print(f"p-valor: {t_test_result.pvalue:.4f}")

            print("\n")
            print("Teste-f (statsmodels):")
            f_test_result = self.result_model.f_test(hypothesis)
            if self.verbose:
                print(f_test_result)
            else:
                print(f"F-estatística: {f_test_result.fvalue[0][0]:.4f}")
                print(f"p-valor: {f_test_result.pvalue:.4f}")

            print("\n")
        except Exception as e:
            print(f"Erro ao validar com statsmodels: {str(e)}")
            raise

    def plot_graph(self):
        """Create and display partial regression plots with enhanced styling."""
        self._print_section("Plotando o gráfico...\n")

        if self.result_model is None:
            print(
                "Modelo statsmodels não disponível. Execute validate_with_statsmodels() primeiro."
            )
            return

        # Set up visualization style
        sns.set_theme(style="whitegrid", palette="viridis", font="DejaVu Sans")

        # Configure figure settings
        plt.rc("figure", figsize=(16, 9))
        plt.rc("font", size=14, family="sans-serif", weight="medium")
        plt.rc(
            "axes", labelsize=14, titlesize=16, titleweight="bold", labelweight="medium"
        )
        plt.rc("lines", linewidth=2.5, markersize=8)
        plt.rc("xtick", labelsize=12)
        plt.rc("ytick", labelsize=12)
        plt.rc("legend", fontsize=12, frameon=True, framealpha=0.7)

        # Create partial regression plots
        fig = sm.graphics.plot_partregress_grid(
            self.result_model,
            grid=(2, 2),
            fig=plt.figure(figsize=(16, 9), dpi=100),
        )

        # Enhance plot styling
        for ax in fig.axes:
            ax.spines["right"].set_visible(False)
            ax.spines["top"].set_visible(False)
            ax.grid(True, linestyle="--", alpha=0.7)

            if ax.get_xlabel():
                ax.set_xlabel(ax.get_xlabel(), fontweight="bold")
            if ax.get_ylabel():
                ax.set_ylabel(ax.get_ylabel(), fontweight="bold")

        # Add title and model information
        model_type = (
            "Log-Linear" if self.loglin else "Log-Log" if self.loglog else "Linear"
        )
        fig.suptitle(
            f"Regressão Linear Múltipla - {model_type}\n",
            fontsize=18,
            fontweight="bold",
        )

        plt.figtext(
            0.5,
            0.01,
            f"Model: {self.dependent_var} vs {self.first_independent_var} and {self.second_independent_var} | R²: {self.result_model.rsquared:.3f}",
            ha="center",
            fontsize=12,
            fontstyle="italic",
        )

        # Apply tight layout with spacing
        fig.tight_layout(rect=[0, 0.03, 1, 0.97], pad=2.0)
        plt.show()
        print("-------------\n")

    def run_analysis(self):
        """Run the complete regression analysis pipeline."""
        try:
            self.initial_data()
            self.transform_data()
            self.calculate_vars_matrices()
            self.transposed_independent_vars_matrix = self.matrix_operation(
                "transpose",
                self.independent_vars_matrix,
                message="Calculando a matriz de variáveis independentes transposta...",
            )
            self.calculate_coefficients()
            self.show_model()
            self.calculate_residuals_and_s2()
            self.calculate_variance_covariance_matrix()
            self.validate_with_statsmodels()
            self.plot_graph()
            return True
        except Exception as e:
            print(f"Erro durante a análise: {str(e)}")
            return False


# Example usage
if __name__ == "__main__":
    data = {
        "Preço da Ação": [21.0, 23.5, 14.6, 13.1, 20.0, 18.1, 20.2, 18.5],
        "Gasto dos Pares": [35, 28, 60, 71, 44, 45, 58, 44],
        "Habilidade do CFO": [5.0, 6.2, 9.5, 8.0, 7.9, 7.0, 8.5, 6.8],
    }
    model_verbose = LinearRegressionModel(
        data,
        "Preço da Ação",
        "Gasto dos Pares",
        "Habilidade do CFO",
        loglin=True,
        verbose=True,
    )
    model_verbose.run_analysis()
