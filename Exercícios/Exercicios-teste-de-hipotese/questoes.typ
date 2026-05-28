#import "@preview/grape-suite:3.1.0": exercise
#import exercise: project, task, subtask

#show: project.with(no: 1,
    title: "Modelos com logaritmos e Testes de hipótese",

    university: [Universidade Federal de Pernambuco],
    institute: [Departamento de Ciências Contábeis e Atuariais],
    seminar: [Contabilometria],

    abstract: "Resolva as questões a seguir, utilize 3 casas decimais para aproximação dos resultados" ,
    show-outline: true,

    author: "Pedro Schuler",

    show-solutions: false,
    show-hints: false,
    
    task-type: [Questão],

    date: datetime(year: 2025, month: 12, day: 1)
)

#let task = task.with(numbering-format: (..n) => numbering("1", ..n))
#let subtask = subtask.with(markers: ("a)", "1)"))

#task[Considerando as informações abaixo, calcule e interprete os coeficientes da regressão linear proposta][
    Você obteve dados de consumo de famílias em função da renda mensal. Considere que variável dependente precisa ser log-linearizada para melhor ajuste ao modelo. Além disso você suspeita que as famílias em que ambos os conjuges possuem carteira de trabalho assinada tem um consumo maior. Você deseja entender o impacto da renda sobre o consumo contrtolando para o efeito do emprego dos conjuges. Interprete ambos os coeficientes e indique o coeficiente do parâmetro de interesse.

    #table(columns: 3,
      table.header[*Consumo (Reais)*][*Renda Mensal (Reais)*][*Emprego Conjuges*],
      [1450],[2300],[Não],
      [2300],[3500],[Não],
      [2000],[3000],[Sim],
      [4200],[5000],[Sim],
      [3100],[4500],[Sim],
      [1100],[2100],[Não]
    )
]

#task[Utilizando os dados da regressão anterior faça o que se pede][
    Você deseja entender o impacto da utilização da variável de controle no seu modelo. Para isso você omitiu a variável de controle e calculou o viés da variável omitida no parâmetro de interesse. Indique a direção do viés.
]

#task[Realize o teste de hipótese definido abaixo.][
    Você deseja testar se o coeficiente da variável de controle é estatisticamente diferente de zero. Utilize um nível de significância de 5% e realize o teste t e utilize t=2,15. Apresente os passos do teste e a conclusão.
]
