%% ======================================================================
% ufpe-exercise.cls - Document class for UFPE exercises sheets
% ======================================================================
% Classe para formatação de questionários e folhas de questões
% Versão 1.0 - Abril 2026
% ======================================================================

\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{ufpe-exercises}[Folhas de exercícios para a UFPE]

%% ======================================================================
% DEFINE OPÇÕES E DEFAULTS DISPONÍVEIS
% ======================================================================
%\DeclareKeys{
%  font.store = \my@font,
%  font.default = 12pt,
%}

%\ProcessKeyOptions

%% ======================================================================
% CARREGAMENTO DE CLASSE BASE
% ======================================================================
\LoadClass[12pt,answers,addpoints]{exam}

% ======================================================================
% CARREGAMENTO DE PACOTES
% ======================================================================
% Pacotes essenciais e de compatibilidade
\RequirePackage{luatex85}

% Pacotes para idioma e tipografia
\RequirePackage{fontspec}
\RequirePackage[brazilian]{babel}
\RequirePackage{microtype}

% Pacotes matemáticos
\RequirePackage{amsmath,amssymb,mathtools}

% Pacotes para citações
\RequirePackage{csquotes}

% Pacotes auxiliares
\RequirePackage{xcolor}
\RequirePackage{ninecolors}
\RequirePackage{tabularray}
\RequirePackage{graphicx}
%\RequirePackage{booktabs}
\UseTblrLibrary{booktabs}

% ======================================================================
% CONFIGURAÇÃO DE FONTES
% ======================================================================
% Configura as fontes padrão conforme normas ABNT
\setmainfont{Times New Roman}
\setsansfont{Arial}
\setmonofont{Courier New}

% ======================================================================
% DEFINIÇÕES DE COMANDOS DA CLASSE
% ======================================================================
\newcommand{\curso}[1]{\def\@CourseName{#1}}
\newcommand{\discente}[1]{\def\@AuthorName{#1}}
\newcommand{\centro}[1]{\def\@CenterName{#1}}
\newcommand{\departamento}[1]{\def\@DepartmentName{#1}}
\newcommand{\titulo}[1]{\def\@TheTitle{#1}}
\newcommand{\disciplina}[1]{\def\@Subject{#1}}
\newcommand{\docente}[1]{\def\@Teacher{#1}}
\newcommand{\periodo}[1]{\def\@Semester{#1}}
\newcommand{\corcabecalho}[1]{\def\@Color{#1}}

% ======================================================================
% FORMATAÇÃO DE SEÇÕES E TÍTULOS
% ======================================================================
\renewcommand{\solutiontitle}{\noindent{\color{\@Color}\textbf{Resposta:}}\enspace}
\qformat{{\color{\@Color}\textbf{Questão \thequestion:}\hfill\\}}

\pointsdroppedatright % suprime a exibição de pontos, exibindo apenas as questões contendo \droppoints 
\unframedsolutions

% ======================================================================
% CONFIGURAÇÃO DE CABEÇALHOS E RODAPÉS
% ======================================================================
\pagestyle{headandfoot}
\cfoot{}

\newcommand{\cabecalho}{
	\begin{minipage}{0.2\textwidth}
		\includegraphics[scale=.25]{logo-universidade.png}
	\end{minipage}
	\hfill
	\begin{minipage}{0.75\textwidth}
		\centering
		{\Large\textbf{Universidade Federal de Pernambuco\\\@CenterName\\\@DepartmentName}}

	\end{minipage}

	\begin{center}
		\begin{tblr}{width=\textwidth,colspec={X[l,m]X[c]X[r,m]},hlines,vlines,row{1} = {abovesep+=15pt,belowsep+=15pt,bg=\@Color}}
			\SetCell[c=3]{c,m} \LARGE\textbf{\@TheTitle}   &  &                              \\
			\SetCell[c=2]{} \textbf{Disciplina:} \@Subject &  & \textbf{Curso:} \@CourseName \\
			\SetCell[c=2]{} \textbf{Docente:} \@Teacher    &  & \textbf{Período:} \@Semester \\
			\SetCell[c=2]{} \textbf{Aluno:} \@AuthorName   &  & \textbf{Data:} \today        \\
		\end{tblr}
	\end{center}

	\vspace{0.5cm}
}
