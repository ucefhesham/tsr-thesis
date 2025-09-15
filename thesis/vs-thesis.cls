% VS Thesis Template based on smart-thesis by Jan Philip Göpfert, Andreas Stöckel
% https://github.com/astoeckel/smart-thesis
\ProvidesClass{vs-thesis}[2024/09/16 VS Thesis]

\RequirePackage{ifthen}
\RequirePackage{ifpdf}
\RequirePackage{ifxetex}
\RequirePackage{ifluatex}

\newboolean{metafont}\setboolean{metafont}{false}
\newboolean{metaprofont}\setboolean{metaprofont}{false}
\newboolean{lesscolor}\setboolean{lesscolor}{false}
\newboolean{twoside}\setboolean{twoside}{false}
\DeclareOption{metafont}{\setboolean{metafont}{true}}
\DeclareOption{metaprofont}{\setboolean{metaprofont}{true}}
\DeclareOption{lesscolor}{\setboolean{lesscolor}{true}}
\DeclareOption{twoside}{\setboolean{twoside}{true}}
\ProcessOptions*

\ifthenelse{\boolean{twoside}}{%
    \LoadClass[a4paper,11pt]{memoir}
}{%
    \LoadClass[a4paper,11pt,oneside]{memoir}
}

% Provides math enviroments, math symbols and more things useful for math.
\RequirePackage{amsmath}
\RequirePackage{amssymb}
\RequirePackage{amsthm}
\RequirePackage{amscd}

% Enhances amsmath and provides further mathematical tools.
\RequirePackage{mathtools}

% Use the following code for XeLaTeX
\ifxetex
    \RequirePackage[expansion=false,final]{microtype}

    \RequirePackage[bigdelims,vvarbb]{newpxmath}
    \RequirePackage[no-math]{fontspec}
    \RequirePackage{polyglossia}
    \RequirePackage{FiraSans}

    \newfontfamily\widespaced[LetterSpace=14, Ligatures=TeX, SmallCapsFeatures={Letters=SmallCaps}, WordSpace=1.8]{cochineal}
    \DeclareRobustCommand{\spacedlowsmallcaps}[1]{{\widespaced\oldstylenums{\textsc{\MakeTextLowercase{#1}}}}}

    \newfontfamily\widerspaced[LetterSpace=18, Ligatures=TeX, SmallCapsFeatures={Letters=SmallCaps}, WordSpace=1.8]{cochineal}
    \DeclareRobustCommand{\spacedallcaps}[1]{{\widerspaced\oldstylenums{\MakeTextUppercase{#1}}}}

    \newcommand{\deen}[2]{\def\german{german}\ifx\xpg@main@language\german#1\else#2\fi}

    \setmainfont{Cochineal}[
        Extension=.otf,
        UprightFont=*-Roman,
        ItalicFont=*-Italic,
        BoldFont=*-Bold,
        BoldItalicFont=*-BoldItalic,
        Numbers={OldStyle}
    ]

    \let\setpdefaultlanguage\setdefaultlanguage
    \renewcommand{\setdefaultlanguage}[1]{
        \setpdefaultlanguage{#1}
        \RequirePackage[noabbrev,capitalize,nameinlink]{cleveref} % Names references automatically. Load after hyperref.
    }
\else
    % Use the following code for LuaTeX
    \ifluatex
        \RequirePackage[
            expansion=false,
            final,
            tracking=true
        ]{microtype}

        \RequirePackage[bigdelims,vvarbb]{newpxmath}
        \RequirePackage[no-math]{fontspec}
        \RequirePackage{polyglossia}
        \RequirePackage{FiraSans}
        \RequirePackage[osf]{cochineal}

        %\newfontfamily\widespaced[LetterSpace=14, Ligatures=TeX, SmallCapsFeatures={Letters=SmallCaps}, WordSpace=1.8]{cochineal}
        %\DeclareRobustCommand{\spacedlowsmallcaps}[1]{{\widespaced\oldstylenums{\textsc{\MakeTextLowercase{#1}}}}}

        %\newfontfamily\widerspaced[LetterSpace=18, Ligatures=TeX, SmallCapsFeatures={Letters=SmallCaps}, WordSpace=1.8]{cochineal}
        %\DeclareRobustCommand{\spacedallcaps}[1]{{\widerspaced\oldstylenums{\MakeTextUppercase{#1}}}}

        \DeclareRobustCommand{\spacedlowsmallcaps}[1]{{\oldstylenums{\textsc{\MakeTextLowercase{#1}}}}}
        \DeclareRobustCommand{\spacedallcaps}[1]{{\oldstylenums{\MakeTextUppercase{#1}}}}

        \newcommand{\deen}[2]{\def\german{german}\ifx\xpg@main@language\german#1\else#2\fi}

        %\setmainfont{Cochineal}[
        %    Extension=.otf,
        %    UprightFont=*-Roman,
        %    ItalicFont=*-Italic,
        %    BoldFont=*-Bold,
        %    BoldItalicFont=*-BoldItalic,
        %    Numbers={OldStyle}
        %]

        \let\setpdefaultlanguage\setdefaultlanguage
        \renewcommand{\setdefaultlanguage}[1]{
            \setpdefaultlanguage{#1}
            \RequirePackage[noabbrev,capitalize,nameinlink]{cleveref} % Names references automatically. Load after hyperref.
        }
    \else
        % Use the following code for pdfLaTex
        \ifpdf
            \RequirePackage[tracking=true,kerning=true,spacing=nonfrench]{microtype}
            \RequirePackage[bigdelims,vvarbb]{newpxmath}
            \RequirePackage[osf]{cochineal}
            \RequirePackage{FiraSans}

            \DeclareRobustCommand{\spacedallcaps}[1]{\textls[160]{\textsc{\MakeTextUppercase{#1}}}}%
            \DeclareRobustCommand{\spacedlowsmallcaps}[1]{\textls[80]{\textsc{\MakeTextLowercase{#1}}}}%
            \RequirePackage[utf8]{inputenc}
            \RequirePackage[T1]{fontenc}
            \RequirePackage{iflang}

            \newcommand{\deen}[2]{\IfLanguageName{ngerman}{#1}{#2}}
            \newcommand{\setdefaultlanguage}[1]{
                \ifthenelse{\equal{#1}{german}}{
                    \usepackage[english,german,ngerman]{babel}
                    \RequirePackage[noabbrev,capitalize,nameinlink,german]{cleveref} % Names references automatically. Load after hyperref.
                }{
                    \usepackage[ngerman,german,english]{babel}
                    \RequirePackage[noabbrev,capitalize,nameinlink,english]{cleveref} % Names references automatically. Load after hyperref.
                }
            }
        \fi
    \fi
\fi


% Context sensitive quotation facilities. Recommended by babel and should be loaded before babel.
\RequirePackage{csquotes}

% Provides inclusion of graphics and a proper interface for '\includegraphics'.
\RequirePackage{graphicx}

% Provides control over list environments. Supersedes 'enumerate', which gives enumerate environment an optional argument which determines the style in which the counter is printed.
\RequirePackage[inline]{enumitem}

% Ugly hack that sometimes helps with otherwise missing spaces.
\RequirePackage{xspace}

% Select MetaPro font
\ifthenelse{\boolean{metaprofont}}{
    \ifxetex
        \setsansfont[%
            Mapping=tex-text,Numbers={OldStyle,Proportional},Scale=.96,
            BoldFont={MetaPro-Bold},
            BoldItalicFont={MetaPro-BoldItalic},
            ItalicFont={MetaPro-BookItalic}
        ]{MetaPro-Book}
    \else
        \ClassError{vs-thesis}{Option metaprofont requires compilation with XeLaTeX}{}
    \fi
}{}

% Select Meta font
\ifthenelse{\boolean{metafont}}{
    \ifxetex
        \setsansfont[%
            Mapping=tex-text,Scale=.96,
            BoldFont={MetaBold-Roman},
            BoldItalicFont={MetaBold-Italic},
            ItalicFont={MetaBook-Italic},
            SmallCapsFont={MetaBook-Caps}
        ]{MetaBook-Roman}
    \else
        \ClassError{vs-thesis}{Option metafont requires compilation with XeLaTeX}{}
    \fi
}{}

\RequirePackage[usenames,dvipsnames,svgnames]{xcolor}
\RequirePackage{ragged2e} % for marginnote
\RequirePackage{ccicons} % for titlepage
\RequirePackage{tikz} % for titlepage
\usetikzlibrary{calc}

% Increase linespread for Palatino
\renewcommand{\linespread}{1.05}

% COLOR DEFINITIONS
\definecolor{uulm}{RGB}{125,154,170}
\definecolor{uulm-akzent}{RGB}{169,162,141}
\definecolor{uulm-in}{RGB}{163,38,56}
\definecolor{uulm-med}{RGB}{38,84,124}
\definecolor{uulm-mawi}{RGB}{86,170,28}
\definecolor{uulm-nawi}{RGB}{189,96,5}

\definecolor{shadecolor}{gray}{0.95}

\colorlet{heading}{uulm-akzent}
\colorlet{cite}{uulm-in}
\colorlet{url}{uulm-in}
\colorlet{ref}{black}
\colorlet{caption}{black}
\colorlet{margincaption}{caption}

%
% SECTIONING STYLES (toledo)
%
% See http://tex.stackexchange.com/questions/161577/memoir-hangnum-chapter-number-in-right-margin-switch for the inspiration
\setsecnumdepth{subsection}

\makeatletter
\makechapterstyle{toledo}{%
    \chapterstyle{default}

    \renewcommand*{\chapterheadstart}{\vspace*{0.0cm}}

    \renewcommand*{\chapnumfont}{\normalfont\color{heading}\addfontfeature{SizeFeatures={Size=110}}}
    \newfont{\chapterNumber}{eurb10 scaled 7000}
    \renewcommand*{\chapnumfont}{\chapterNumber\color{heading}}
    \renewcommand*{\printchaptername}{}
    \renewcommand*{\chapternamenum}{}
    \renewcommand*{\printchapternum}{%
        \raisebox{0pt}[0pt][0pt]{%
        \makebox[0pt][l]{%
        \makebox[\dimexpr\textwidth+7.25em\relax][l]{%
            \parbox[t]{\textwidth}{\mbox{}}%
            \parbox[t]{7.25em}{\hfill\chapnumfont \thechapter}}}}}%
    \renewcommand*{\chapternamenum}{}
    \renewcommand*{\afterchapternum}{}

    \renewcommand*{\afterchaptertitle}{%
    \par\parbox{\textwidth}{\hrulefill}\par%
    \vspace*{4ex}}
    \renewcommand*{\chaptitlefont}{\normalfont\large}
    \renewcommand*{\printchaptertitle}[1]{%
        \parbox[b]{\textwidth}{%
            \raggedright\chaptitlefont\spacedallcaps{##1}%
        }%
    }

    \setsecheadstyle{\spacedlowsmallcaps}
    \setsecindent{-3.5ex}
    \setbeforesecskip{-4ex plus -1ex minus -.2ex}
    \setaftersecskip{4ex plus 1ex minus .2ex}

    \setsubsecheadstyle{\spacedlowsmallcaps}
    \setsubsecindent{-3.5ex}
    \setbeforesubsecskip{-4ex plus -1ex minus -.2ex}
    \setaftersubsecskip{4ex plus 1ex minus .2ex}
    \setsecnumformat{\spacedlowsmallcaps{\csname the##1\endcsname\quad}}
    \renewcommand*{\abstractnamefont}{\spacedlowsmallcaps}
}
\makeatother

\chapterstyle{toledo}

%
% FIGURES
%

\feetbelowfloat
\captionstyle{\small}
\captionnamefont{\slshape\color{caption}}

\newsubfloat{figure}% Create subfloat in figure environment
\newsubfloat{table}% Create subfloat in table environment

%
% ACKNOWLEDGEMENT
%

\makeatletter
\newcommand{\acknowledgementname}{Acknowledgements}
\newenvironment{acknowledgement}{%
  \setup@bstract
  \if@bsrunin\else
    \ifnumber@bs \num@bs \else
      \begin{\absnamepos}\abstractnamefont\acknowledgementname\end\absnamepos%
      \vspace{\abstitleskip}%
    \fi
  \fi
  \put@bsintoc%
  \begin{@bstr@ctlist}\if@bsrunin\@bsrunintitle\fi\abstracttextfont}%
  {\par\end{@bstr@ctlist}}

\newcommand{\initialepigraph}[2]{\begingroup
    \cleardoublepage
    \thispagestyle{empty}
    \vspace*{2cm}
    \epigraph{#1}{#2}
    \cleardoublepage
\endgroup}

\renewcommand{\abstract}[1]{\begingroup
    \cleardoublepage
    \deen{\chapter{Zusammenfassung}}{\chapter{Abstract}}
    #1
    \cleardoublepage
\endgroup}

\let\oldepigraph\epigraph
\renewcommand{\epigraph}[2]{\oldepigraph{#1}{\spacedlowsmallcaps{#2}}}

\makeatother

%
% MARGINNOTE
%

%
% There are three commands for placing elements in the margin:
%
% \marginnote{<TEXT>}
% Just like \marginpar but with correct formating
%
% \marginfig{<SHORT CAPTION>}{<FIGURE CONTENT>}{<LONG CAPTION>}{\label{<LABEL>}}
% \margintbl{<SHORT CAPTION>}{<TABLE CONTENT>}{<LONG CAPTION>}{\label{<LABEL>}}
% Places a figure/table in the margin
% - <SHORT CAPTION> is the caption as it occurs in the list of figures/tables
% - <FIGURE CONTENT>/<TABLE CONTENT> is the actual figure/table content
% - <LONG CAPTION> is the caption below the figure
% - <LABEL> is the label -- we decided to let you explicitly type "\label" here
% as this allows LaTeX editors to index the label and provide auto completion.
%

% Define the "marginnote" command. For compatibility reasons
% we do not override marginpar.
\strictpagecheck
\newcommand{\marginnote}[1]{\hspace*{0pt}\marginpar{
    \slshape\footnotesize%
    \vspace*{-0.725em}%
    \checkoddpage%
    \ifoddpage%
        \RaggedRight%
    \else%
        \RaggedLeft%
    \fi%
    {#1}%
}\ignorespaces}

\newcommand{\marginfloat}[8]{\marginnote{{#1}%
\par\refstepcounter{#4}\textcolor{margincaption}{#5 #6}: {#2}#3}%
\addcontentsline{#7}{#4}{\protect\numberline {#6}{\ignorespaces #8}}\ignorespaces}

\newcommand{\marginfig}[4]{\marginfloat{#2}{#3}{#4}{figure}{Figure}{\thefigure}{lof}{#1}}
\newcommand{\margintbl}[4]{\marginfloat{#2}{#3}{#4}{table}{Table}{\thetable}{lot}{#1}}

%
% PAGE STYLE (headers, footers)
%

\makeatletter

% Deactivate capitalization of headers.
\nouppercaseheads

% Define the new "berlin" style.
\makepagestyle{berlin}

% Display the page number in the footer.
\makeevenfoot{berlin}{\thepage}{}{}
\makeoddfoot{berlin}{}{}{\thepage}

% Override the chapter style with the unfinished "berlin" style
% to eliminate the header and to make sure a consistent footer
% style is used.
\copypagestyle{chapter}{berlin}

% Define the "berlin" style
\makepsmarks{berlin}{%
    \def\chaptermark##1{%
    \markboth{\memUChead{%
        \ifnum \c@secnumdepth >\m@ne
        \if@mainmatter
            \@chapapp\ \thechapter. \ %
        \fi
        \fi
        ##1}}{}}%
    \def\tocmark{\markboth{\memUChead{\contentsname}}{\memUChead{\contentsname}}}%
    \def\lofmark{\markboth{\memUChead{\listfigurename}}{\memUChead{\listfigurename}}}%
    \def\lotmark{\markboth{\memUChead{\listtablename}}{\memUChead{\listtablename}}}%
    \def\bibmark{\markboth{\memUChead{\bibname}}{\memUChead{\bibname}}}%
    \def\indexmark{\markboth{\memUChead{\indexname}}{\memUChead{\indexname}}}%
    \def\sectionmark##1{%
    \markright{\memUChead{%
        \ifnum \c@secnumdepth > \z@
        \thesection. \ %
        \fi %
        ##1}}}}
\makepsmarks{berlin}{%
    \createmark{chapter}{left}{nonumber}{}{}
    \createmark{section}{right}{shownumber}{}{\ }
    \createplainmark{toc}{both}{\contentsname}
    \createplainmark{lof}{both}{\listfigurename}
    \createplainmark{lot}{both}{\listtablename}
    \createplainmark{bib}{both}{\bibname}
    \createplainmark{index}{both}{\indexname}
    \createplainmark{glossary}{both}{\glossaryname}
}

% Discriminate between "oneside" and "twoside" documents.
\if@twoside
    \makeevenhead{berlin}{\spacedlowsmallcaps{\leftmark}}{}{}
    \makeoddhead{berlin}{}{}{\spacedlowsmallcaps{\rightmark}}
\else
    \makeoddhead{berlin}{\spacedlowsmallcaps{\rightmark}}{}{}
\fi

% Use the berlin style.
\pagestyle{berlin}

% style part header pages
\renewcommand{\printpartname}{}
\newfont{\partNumber}{eurb10 scaled 7000}
\renewcommand{\partnumfont}{\partNumber\color{uulm-in}\raggedleft}

\makeatother

%
% FOOTNOTES
%

% Fix memoir bug, otherwise the symbol is small in the text and small capitalized in the reference
\makeatletter
\renewcommand*{\@makefnmark}{\hbox{\textsuperscript{\textsc{\@thefnmark}}}}
\footmarkstyle{\textsc{#1}}
\setlength{\footmarkwidth}{-0.5em}
\setlength{\footmarksep}{0.5em}
\setlength{\footparindent}{0em}

%
% EPIGRAPHS
%
\makeatletter

% Make the epigraph span two third of the text width
\setlength{\epigraphwidth}{0.66\textwidth}

% Disable the rule below the epigraph
\setlength{\epigraphrule}{0pt}

% Use the normal font size
\epigraphfontsize{\normalsize}

% Style the epigraph itself -- use italic text and
% flush it to the right
\newenvironment{epigraphstyle}{
    \small\slshape%
%    \RaggedLeft%
}{%
}
\epigraphtextposition{epigraphstyle}

% Style the epigraph source -- prepend it with a em-dash
% and make sure no indentation is added after the epigraph
\newenvironment{epigraphsourcestyle}%
{%#
    --- \small\raggedleft\color{uulm-in}%
}{%
    \@afterindentfalse\@afterheading%
}
\epigraphsourceposition{epigraphsourcestyle}

\makeatother

%
% TITLEPAGE
%

\makeatletter

\newcommand\titlegraphic[1]{\renewcommand\@titlegraphic{%
    \node[inner sep=0, anchor=south west] at ($ (current page.south west) + (0, 17.7cm) $) {%
        \includegraphics[keepaspectratio, width=\paperwidth, height=6.8cm]{#1}};}}
\newcommand\@titlegraphic{}

\newcommand\thesistype[1]{\renewcommand\@thesistype{#1}}
\newcommand\@thesistype{}

\newcommand\authorprefix[1]{\renewcommand\@authorprefix{#1}}
\newcommand\@authorprefix{}

\newcommand\studentnumber[1]{\renewcommand\@studentnumber{#1}}
\newcommand\@studentnumber{}

\newcommand\attributions[1]{\renewcommand\@attributions{#1}}
\newcommand\@attributions{}

\newcommand\institution[1]{\renewcommand\@institution{#1}}
\newcommand\@institution{}

\newcommand\supervisors[1]{\renewcommand\@supervisors{#1}}
\newcommand\@supervisors{}

\newcommand\examiners[1]{\renewcommand\@examiners{#1}}
\newcommand\@examiners{}

\newcommand\internalid[1]{\renewcommand\@internalid{#1}}
\newcommand\@internalid{}

\newcommand\license[1]{\renewcommand\@license{#1}}
\newcommand\@license{\ccCopy}

\newcommand\submissiondate[1]{\renewcommand\@submissiondate{#1}}
\newcommand\@submissiondate{}

\newcommand\place[1]{\renewcommand\@place{#1}}
\newcommand\@place{}

\newcommand\independencedeclaration[1]{\renewcommand\@independencedeclaration{#1}}
\newcommand\@independencedeclaration{{\begingroup%
    \pagebreak%
    \thispagestyle{empty}%
    \par\vspace*{\fill}%
    \noindent%
    \deen{
        Hiermit versichere ich, dass ich die vorliegende Arbeit mit dem Titel
    }{
        I hereby declare that this thesis titled:
    }
    \begin{center}
      \textbf{\@title}
    \end{center}
    \deen{
        bis auf die offizielle Betreuung selbstständig und ohne fremde Hilfe angefertigt habe und die benutzten Quellen und Hilfsmittel vollständig angegeben sind.
        Aus fremden Quellen direkt oder indirekt übernommenen Gedanken sind jeweils unter Angabe der Quelle als solche kenntlich gemacht.\\
        Ich erkläre außerdem, dass die vorliegende Arbeit entsprechend den Grundsätzen guten wissenschaftlichen Arbeitens gemäß der „Satzung der Universität Ulm zur Sicherung guter wissenschaftlicher Praxis“ erstellt wurde.
    }{
        is the product of my own independent work and that I have used no sources or materials other than those specified.
        The passages taken from other works, either verbatim or paraphrased in the spirit of the original quote, are identified in each individual case by indicating the source.\\
        I further declare that all my academic work was written in line with the principles of proper academic research according to the official ``Satzung der Universität Ulm zur Sicherung guter wissenschaftlicher Praxis'' (University Statute for the Safeguarding of Proper Academic Practice).
    }

    \begin{flushleft}
      \@place, \deen{den}{} \@submissiondate

      \vspace{2cm} \hspace{1cm} \@author, \deen{Matrikelnummer}{student number} \@studentnumber\\
    \end{flushleft}

    \vspace{2cm}
    \endgroup}}


\newcommand\uulmlogo[3]{%{{{
    \begin{tikzpicture}[y=0.8pt, x=0.8pt, inner sep=0pt, outer sep=0pt]
        \pgftransformscale{#3/135mm}
        \begin{scope}[cm={{1.25,0.0,0.0,-1.25,(0.0,141.7325)}}]
        \path[fill=#2] (152.1000,29.3000) -- (141.0000,29.3000) -- (141.0000,50.2000) .. controls (141.0000,55.7000) and (139.3000,64.4000) .. (131.2000,64.4000) .. controls (124.1000,64.4000) and (124.0000,57.3000) .. (124.0000,52.1000) -- (124.0000,29.3000) -- (112.9000,29.3000) -- (112.9000,56.3000) .. controls (112.9000,65.9000) and (117.3000,73.1000) .. (127.4000,73.1000) .. controls (133.2000,73.1000) and (138.0000,70.4000) .. (141.4000,66.3000) -- (141.6000,66.3000) -- (141.6000,72.1000) -- (152.1000,72.1000) -- (152.1000,29.3000) -- cycle;

        \path[fill=#2,rounded corners=0.0000cm] (210.5000,7.8000) rectangle (215.7000,72.1000);

        \path[fill=#2] (199.2000,11.2000) -- (196.9000,11.2000) -- (196.9000,8.3000) -- (195.6000,8.7000) -- (195.6000,11.1000) -- (193.6000,11.1000) -- (193.6000,12.2000) -- (195.6000,12.2000) -- (195.6000,18.1000) .. controls (195.6000,20.0000) and (195.7000,21.5000) .. (197.9000,21.5000) .. controls (198.5000,21.5000) and (199.1000,21.3000) .. (199.4000,21.2000) -- (199.3000,20.2000) .. controls (198.9000,20.3000) and (198.6000,20.4000) .. (198.1000,20.4000) .. controls (197.1000,20.4000) and (196.7000,19.6000) .. (196.7000,18.7000) -- (196.7000,12.1000) -- (199.0000,12.1000) -- (199.0000,11.2000) -- cycle(185.7000,9.5000) -- (187.2000,9.5000) -- (187.2000,7.8000) -- (185.7000,7.8000) -- (185.7000,9.5000) -- cycle(189.1000,9.5000) -- (190.6000,9.5000) -- (190.6000,7.8000) -- (189.1000,7.8000) -- (189.1000,9.5000) -- cycle(190.5000,16.8000) .. controls (190.5000,17.9000) and (190.4000,20.5000) .. (187.5000,20.5000) .. controls (186.3000,20.5000) and (185.4000,20.0000) .. (185.4000,18.7000) .. controls (185.4000,16.5000) and (188.0000,16.3000) .. (190.2000,16.3000) -- (190.6000,16.3000) -- (190.6000,16.8000) -- cycle(190.6000,21.3000) -- (191.8000,21.3000) .. controls (191.7000,20.6000) and (191.7000,20.0000) .. (191.7000,19.2000) -- (191.7000,14.7000) .. controls (191.7000,12.1000) and (190.6000,10.9000) .. (188.1000,10.9000) .. controls (187.2000,10.9000) and (185.9000,11.2000) .. (185.1000,11.6000) -- (185.1000,12.8000) .. controls (186.0000,12.2000) and (187.0000,12.0000) .. (188.1000,12.0000) .. controls (189.8000,12.0000) and (190.4000,12.9000) .. (190.4000,14.6000) -- (190.4000,15.1000) -- (190.2000,15.1000) .. controls (187.6000,15.1000) and (184.0000,15.3000) .. (184.0000,18.7000) .. controls (184.0000,19.6000) and (184.5000,21.6000) .. (187.3000,21.6000) .. controls (188.6000,21.6000) and (190.0000,20.9000) .. (190.5000,19.7000) -- (190.5000,19.7000) -- (190.5000,21.3000) -- cycle(182.4000,11.2000) -- (180.1000,11.2000) -- (180.1000,8.3000) -- (178.8000,8.7000) -- (178.8000,11.1000) -- (176.8000,11.1000) -- (176.8000,12.2000) -- (178.8000,12.2000) -- (178.8000,18.1000) .. controls (178.8000,20.0000) and (178.9000,21.5000) .. (181.1000,21.5000) .. controls (181.7000,21.5000) and (182.3000,21.3000) .. (182.6000,21.2000) -- (182.5000,20.2000) .. controls (182.1000,20.3000) and (181.8000,20.4000) .. (181.3000,20.4000) .. controls (180.3000,20.4000) and (179.9000,19.6000) .. (179.9000,18.7000) -- (179.9000,12.1000) -- (182.2000,12.1000) -- (182.2000,11.2000) -- cycle(174.7000,6.8000) -- (173.4000,6.8000) -- (173.4000,8.5000) -- (174.7000,8.5000) -- (174.7000,6.8000) -- cycle(173.5000,21.3000) -- (174.8000,21.3000) -- (174.8000,11.2000) -- (173.5000,11.2000) -- (173.5000,21.3000) -- cycle(164.7000,21.1000) .. controls (165.6000,21.5000) and (166.6000,21.6000) .. (167.5000,21.6000) .. controls (169.0000,21.6000) and (171.0000,20.9000) .. (171.0000,18.7000) .. controls (171.0000,15.4000) and (166.0000,15.6000) .. (166.0000,13.7000) .. controls (166.0000,12.4000) and (166.9000,12.0000) .. (168.2000,12.0000) .. controls (168.9000,12.0000) and (169.8000,12.2000) .. (170.4000,12.5000) -- (170.5000,11.4000) .. controls (169.6000,11.1000) and (168.5000,10.9000) .. (168.0000,10.9000) .. controls (166.3000,10.9000) and (164.7000,11.7000) .. (164.7000,13.7000) .. controls (164.7000,16.6000) and (169.6000,16.3000) .. (169.6000,18.7000) .. controls (169.6000,19.9000) and (168.5000,20.5000) .. (167.4000,20.5000) .. controls (166.4000,20.5000) and (165.5000,20.2000) .. (164.8000,19.9000) -- (164.7000,21.1000) -- cycle(158.9000,21.3000) -- (160.2000,21.3000) -- (160.2000,16.3000) .. controls (160.2000,14.5000) and (160.7000,12.2000) .. (162.7000,12.2000) .. controls (163.0000,12.2000) and (163.4000,12.2000) .. (163.6000,12.3000) -- (163.6000,11.0000) .. controls (163.4000,10.9000) and (163.0000,10.9000) .. (162.7000,10.9000) .. controls (161.4000,10.9000) and (160.6000,12.0000) .. (160.1000,13.1000) -- (160.1000,13.1000) -- (160.1000,11.1000) -- (158.8000,11.1000) .. controls (158.9000,11.8000) and (158.9000,12.3000) .. (158.9000,13.4000) -- (158.9000,21.3000) -- cycle(149.4000,15.6000) .. controls (149.4000,13.9000) and (150.5000,12.1000) .. (152.4000,12.1000) .. controls (154.3000,12.1000) and (155.0000,13.9000) .. (155.0000,15.6000) -- (149.4000,15.6000) -- cycle(155.6000,19.8000) .. controls (155.0000,20.2000) and (153.7000,20.6000) .. (152.8000,20.6000) .. controls (150.6000,20.6000) and (149.5000,18.9000) .. (149.5000,16.7000) -- (156.4000,16.7000) -- (156.4000,16.0000) .. controls (156.4000,13.3000) and (155.2000,11.0000) .. (152.4000,11.0000) .. controls (149.8000,11.0000) and (148.1000,13.2000) .. (148.1000,16.3000) .. controls (148.1000,19.3000) and (149.4000,21.6000) .. (152.7000,21.6000) .. controls (153.6000,21.6000) and (154.8000,21.4000) .. (155.6000,21.0000) -- (155.6000,19.8000) -- cycle(146.9000,11.2000) -- (145.6000,11.2000) -- (142.7000,20.0000) -- (142.7000,20.0000) -- (139.9000,11.2000) -- (138.5000,11.2000) -- (141.9000,21.3000) -- (143.4000,21.3000) -- (146.9000,11.2000) -- cycle(136.6000,6.8000) -- (135.3000,6.8000) -- (135.3000,8.5000) -- (136.6000,8.5000) -- (136.6000,6.8000) -- cycle(135.3000,21.3000) -- (136.6000,21.3000) -- (136.6000,11.2000) -- (135.3000,11.2000) -- (135.3000,21.3000) -- cycle(124.1000,21.3000) -- (125.4000,21.3000) -- (125.4000,15.7000) .. controls (125.4000,14.0000) and (126.1000,12.0000) .. (128.4000,12.0000) .. controls (130.1000,12.0000) and (130.8000,13.2000) .. (130.8000,15.1000) -- (130.8000,21.3000) -- (132.1000,21.3000) -- (132.1000,14.8000) .. controls (132.1000,12.6000) and (131.1000,10.9000) .. (128.6000,10.9000) .. controls (126.5000,10.9000) and (125.7000,12.0000) .. (125.3000,13.0000) -- (125.3000,13.0000) -- (125.3000,11.2000) -- (124.1000,11.2000) .. controls (124.2000,12.0000) and (124.2000,12.8000) .. (124.2000,13.6000) -- (124.2000,21.3000) -- cycle(120.9000,11.2000) -- (119.6000,11.2000) -- (119.6000,16.8000) .. controls (119.6000,18.5000) and (118.9000,20.5000) .. (116.6000,20.5000) .. controls (114.9000,20.5000) and (114.2000,19.3000) .. (114.2000,17.4000) -- (114.2000,11.2000) -- (113.0000,11.2000) -- (113.0000,17.7000) .. controls (113.0000,19.9000) and (114.0000,21.6000) .. (116.5000,21.6000) .. controls (118.6000,21.6000) and (119.4000,20.5000) .. (119.8000,19.6000) -- (119.8000,19.6000) -- (119.8000,21.4000) -- (121.0000,21.4000) .. controls (120.9000,20.6000) and (120.9000,19.8000) .. (120.9000,19.0000) -- (120.9000,11.2000) -- cycle;

        \path[fill=#2] (198.6000,29.2000) -- (193.3000,29.2000) -- (193.3000,52.9000) .. controls (193.3000,60.0000) and (190.3000,68.5000) .. (180.6000,68.5000) .. controls (173.3000,68.5000) and (170.4000,63.3000) .. (170.4000,55.2000) -- (170.4000,29.2000) -- (165.1000,29.2000) -- (165.1000,56.6000) .. controls (165.1000,66.0000) and (169.1000,73.1000) .. (179.7000,73.1000) .. controls (188.6000,73.1000) and (191.8000,68.3000) .. (193.5000,64.5000) -- (193.7000,64.5000) -- (193.7000,72.0000) -- (198.8000,72.0000) .. controls (198.5000,68.6000) and (198.5000,65.2000) .. (198.5000,61.9000) -- (198.5000,29.2000) -- cycle;

        \path[fill=#2] (283.9000,72.1000) -- (283.9000,42.6000) .. controls (283.9000,34.3000) and (279.0000,28.3000) .. (270.4000,28.3000) .. controls (263.7000,28.3000) and (258.7000,32.3000) .. (256.7000,37.4000) -- (256.5000,37.4000) .. controls (255.2000,29.9000) and (249.0000,28.3000) .. (245.2000,28.3000) .. controls (236.1000,28.3000) and (233.6000,33.7000) .. (231.9000,36.6000) -- (231.7000,36.5000) -- (231.7000,29.3000) -- (226.6000,29.3000) .. controls (226.9000,32.7000) and (226.9000,36.1000) .. (226.9000,39.4000) -- (226.9000,72.1000) -- (232.2000,72.1000) -- (232.2000,48.4000) .. controls (232.2000,37.8000) and (237.9000,32.8000) .. (243.7000,32.8000) .. controls (250.5000,32.8000) and (252.7000,37.6000) .. (252.7000,43.6000) -- (252.7000,72.0000) -- (258.0000,72.0000) -- (258.0000,48.4000) .. controls (258.0000,37.8000) and (263.7000,32.8000) .. (269.5000,32.8000) .. controls (276.3000,32.8000) and (278.5000,37.6000) .. (278.5000,43.6000) -- (278.5000,72.0000) -- (283.9000,72.0000) -- cycle;

        \path[fill=#2] (32.0000,55.3000) .. controls (32.3000,56.7000) and (32.6000,58.0000) .. (33.0000,59.2000) .. controls (33.0000,59.2000) and (33.0000,59.2000) .. (33.1000,59.2000) .. controls (34.2000,59.2000) and (35.2000,60.1000) .. (35.2000,61.3000) .. controls (35.2000,61.9000) and (34.9000,62.5000) .. (34.4000,62.9000) .. controls (36.0000,66.4000) and (37.8000,68.3000) .. (39.7000,68.3000) .. controls (42.6000,68.3000) and (45.7000,63.3000) .. (47.4000,55.0000) .. controls (45.0000,54.2000) and (42.5000,53.1000) .. (40.0000,51.9000) .. controls (37.2000,53.3000) and (34.6000,54.4000) .. (32.0000,55.3000);

        \path[fill=#2] (34.5000,49.1000) .. controls (33.3000,48.4000) and (32.0000,47.6000) .. (30.8000,46.8000) .. controls (31.0000,49.4000) and (31.3000,51.7000) .. (31.7000,53.9000) .. controls (33.8000,53.1000) and (36.0000,52.2000) .. (38.3000,51.2000) .. controls (37.1000,50.5000) and (35.8000,49.8000) .. (34.5000,49.1000);

        \path[fill=#2] (45.0000,31.0000) .. controls (46.2000,31.7000) and (47.3000,32.4000) .. (48.5000,33.1000) .. controls (48.3000,30.8000) and (48.0000,28.6000) .. (47.6000,26.6000) .. controls (45.6000,27.3000) and (43.6000,28.1000) .. (41.5000,29.1000) .. controls (42.6000,29.7000) and (43.8000,30.3000) .. (45.0000,31.0000);

        \path[fill=#2] (24.8000,25.7000) .. controls (23.8000,25.7000) and (23.0000,25.0000) .. (22.8000,24.1000) .. controls (22.0000,24.0000) and (21.3000,24.0000) .. (20.7000,24.0000) .. controls (18.0000,24.0000) and (16.2000,24.7000) .. (15.4000,26.0000) .. controls (14.0000,28.5000) and (16.7000,33.6000) .. (23.0000,39.2000) .. controls (25.0000,37.5000) and (27.1000,35.9000) .. (29.4000,34.3000) .. controls (29.6000,31.4000) and (30.0000,28.5000) .. (30.5000,25.9000) .. controls (29.1000,25.5000) and (27.8000,25.1000) .. (26.5000,24.8000) .. controls (26.2000,25.4000) and (25.5000,25.7000) .. (24.8000,25.7000);

        \path[fill=#2] (41.5000,51.2000) .. controls (43.6000,52.2000) and (45.7000,53.0000) .. (47.6000,53.7000) .. controls (48.0000,51.7000) and (48.3000,49.5000) .. (48.5000,47.1000) .. controls (47.3000,47.9000) and (46.1000,48.6000) .. (44.9000,49.3000) .. controls (43.7000,50.0000) and (42.6000,50.6000) .. (41.5000,51.2000);

        \path[fill=#2] (47.3000,25.2000) .. controls (47.0000,23.6000) and (46.6000,22.1000) .. (46.1000,20.7000) .. controls (45.0000,20.6000) and (44.1000,19.7000) .. (44.1000,18.6000) .. controls (44.1000,18.0000) and (44.3000,17.5000) .. (44.7000,17.2000) .. controls (43.2000,14.0000) and (41.4000,12.2000) .. (39.6000,12.2000) .. controls (36.7000,12.2000) and (33.7000,17.0000) .. (32.0000,25.0000) .. controls (34.5000,25.9000) and (37.2000,27.0000) .. (39.8000,28.3000) .. controls (42.4000,27.1000) and (44.9000,26.0000) .. (47.3000,25.2000);

        \path[fill=#2] (24.0000,40.1000) .. controls (25.6000,41.4000) and (27.3000,42.8000) .. (29.3000,44.1000) .. controls (29.2000,42.8000) and (29.2000,41.5000) .. (29.2000,40.2000) .. controls (29.2000,38.8000) and (29.2000,37.4000) .. (29.3000,36.0000) .. controls (27.4000,37.4000) and (25.6000,38.8000) .. (24.0000,40.1000);

        \path[fill=#2] (38.3000,29.1000) .. controls (36.1000,28.0000) and (33.9000,27.1000) .. (31.7000,26.4000) .. controls (31.3000,28.5000) and (31.0000,30.9000) .. (30.8000,33.4000) .. controls (32.0000,32.6000) and (33.2000,31.9000) .. (34.4000,31.2000) .. controls (35.7000,30.5000) and (37.0000,29.7000) .. (38.3000,29.1000);

        \path[fill=#2] (23.0000,41.0000) .. controls (21.9000,42.0000) and (20.8000,43.0000) .. (19.9000,44.0000) .. controls (20.1000,44.3000) and (20.2000,44.7000) .. (20.2000,45.0000) .. controls (20.2000,46.1000) and (19.3000,47.1000) .. (18.1000,47.1000) .. controls (17.9000,47.1000) and (17.6000,47.1000) .. (17.4000,47.0000) .. controls (15.1000,50.1000) and (14.4000,52.7000) .. (15.3000,54.3000) .. controls (16.4000,56.3000) and (20.1000,56.8000) .. (25.3000,55.7000) .. controls (26.9000,55.4000) and (28.6000,54.9000) .. (30.4000,54.3000) .. controls (29.9000,51.6000) and (29.5000,48.7000) .. (29.3000,45.8000) .. controls (27.1000,44.3000) and (25.0000,42.7000) .. (23.0000,41.0000);

        \path[fill=#2] (36.9000,45.2000) .. controls (36.9000,45.8000) and (37.9000,46.6000) .. (39.4000,46.8000) .. controls (39.4000,46.8000) and (39.4000,46.8000) .. (39.4000,46.8000) -- (39.4000,43.7000) .. controls (38.2000,43.8000) and (36.9000,44.5000) .. (36.9000,45.2000);

        \path[fill=#2] (55.6000,40.1000) .. controls (53.9000,38.7000) and (52.1000,37.2000) .. (50.0000,35.8000) .. controls (50.1000,37.3000) and (50.1000,38.8000) .. (50.1000,40.2000) .. controls (50.1000,41.6000) and (50.1000,43.0000) .. (50.0000,44.4000) .. controls (52.1000,43.0000) and (54.0000,41.5000) .. (55.6000,40.1000);

        \path[fill=#2] (56.6000,41.0000) .. controls (54.6000,42.8000) and (52.3000,44.5000) .. (49.9000,46.1000) .. controls (49.7000,48.9000) and (49.4000,51.6000) .. (48.9000,54.1000) .. controls (50.4000,54.6000) and (51.9000,55.0000) .. (53.4000,55.3000) .. controls (53.7000,54.6000) and (54.4000,54.2000) .. (55.2000,54.2000) .. controls (56.2000,54.2000) and (57.1000,54.9000) .. (57.2000,55.9000) .. controls (60.7000,56.2000) and (63.1000,55.5000) .. (64.0000,54.0000) .. controls (65.5000,51.6000) and (62.8000,46.6000) .. (56.6000,41.0000);

        \path[fill=#2] (58.8000,57.4000) .. controls (58.2000,57.4000) and (57.7000,57.4000) .. (57.1000,57.3000) .. controls (56.7000,57.9000) and (56.1000,58.4000) .. (55.3000,58.4000) .. controls (54.3000,58.4000) and (53.5000,57.7000) .. (53.3000,56.7000) .. controls (51.8000,56.4000) and (50.3000,56.0000) .. (48.8000,55.5000) .. controls (47.1000,63.6000) and (44.0000,69.7000) .. (39.8000,69.7000) .. controls (37.1000,69.7000) and (34.9000,67.2000) .. (33.2000,63.4000) .. controls (33.2000,63.4000) and (33.2000,63.4000) .. (33.2000,63.4000) .. controls (32.1000,63.4000) and (31.1000,62.5000) .. (31.1000,61.3000) .. controls (31.1000,60.7000) and (31.4000,60.1000) .. (31.8000,59.7000) .. controls (31.4000,58.4000) and (31.1000,57.1000) .. (30.8000,55.7000) .. controls (29.0000,56.3000) and (27.3000,56.7000) .. (25.7000,57.1000) .. controls (23.9000,57.5000) and (22.2000,57.7000) .. (20.7000,57.7000) .. controls (17.5000,57.7000) and (15.2000,56.8000) .. (14.2000,55.1000) .. controls (12.9000,52.8000) and (13.9000,49.6000) .. (16.4000,46.2000) .. controls (16.2000,45.9000) and (16.1000,45.5000) .. (16.1000,45.1000) .. controls (16.1000,44.0000) and (17.0000,43.0000) .. (18.2000,43.0000) .. controls (18.5000,43.0000) and (18.7000,43.1000) .. (19.0000,43.2000) .. controls (19.9000,42.2000) and (21.0000,41.2000) .. (22.1000,40.2000) .. controls (15.9000,34.7000) and (12.3000,29.0000) .. (14.3000,25.4000) .. controls (15.5000,23.3000) and (18.6000,22.4000) .. (23.0000,22.9000) .. controls (23.3000,22.2000) and (24.0000,21.7000) .. (24.9000,21.7000) .. controls (26.0000,21.7000) and (26.9000,22.5000) .. (27.0000,23.6000) .. controls (28.3000,23.9000) and (29.6000,24.3000) .. (30.9000,24.7000) .. controls (32.6000,16.8000) and (35.7000,10.9000) .. (39.8000,10.9000) .. controls (42.3000,10.9000) and (44.5000,13.1000) .. (46.1000,16.7000) .. controls (46.2000,16.7000) and (46.3000,16.7000) .. (46.3000,16.7000) .. controls (47.4000,16.7000) and (48.4000,17.6000) .. (48.4000,18.8000) .. controls (48.4000,19.5000) and (48.0000,20.1000) .. (47.5000,20.5000) .. controls (47.9000,21.9000) and (48.3000,23.4000) .. (48.7000,24.9000) .. controls (50.5000,24.3000) and (52.2000,23.9000) .. (53.8000,23.5000) .. controls (59.7000,22.3000) and (63.8000,23.1000) .. (65.3000,25.6000) .. controls (66.5000,27.8000) and (65.7000,30.7000) .. (63.5000,33.8000) .. controls (63.7000,34.1000) and (63.8000,34.5000) .. (63.8000,35.0000) .. controls (63.8000,36.1000) and (62.9000,37.1000) .. (61.7000,37.1000) .. controls (61.4000,37.1000) and (61.2000,37.0000) .. (61.0000,37.0000) .. controls (60.0000,38.1000) and (58.9000,39.2000) .. (57.7000,40.3000) .. controls (63.8000,45.8000) and (67.4000,51.4000) .. (65.3000,55.0000) .. controls (64.3000,56.5000) and (62.0000,57.4000) .. (58.8000,57.4000)(39.7000,8.6000) .. controls (22.2000,8.6000) and (8.0000,22.8000) .. (8.0000,40.2000) .. controls (8.0000,57.7000) and (22.2000,71.8000) .. (39.6000,71.8000) .. controls (57.0000,71.8000) and (71.2000,57.6000) .. (71.2000,40.2000) .. controls (71.3000,22.8000) and (57.1000,8.6000) .. (39.7000,8.6000);

        \path[fill=#2] (56.6000,39.2000) .. controls (57.8000,38.1000) and (58.9000,37.0000) .. (59.9000,35.9000) .. controls (59.7000,35.6000) and (59.6000,35.2000) .. (59.6000,34.8000) .. controls (59.6000,33.7000) and (60.5000,32.7000) .. (61.7000,32.7000) .. controls (61.9000,32.7000) and (62.1000,32.7000) .. (62.3000,32.8000) .. controls (64.2000,29.9000) and (64.8000,27.6000) .. (64.0000,26.1000) .. controls (63.3000,24.8000) and (61.4000,24.1000) .. (58.7000,24.1000) .. controls (57.3000,24.1000) and (55.7000,24.3000) .. (54.0000,24.6000) .. controls (52.4000,24.9000) and (50.7000,25.4000) .. (48.9000,26.0000) .. controls (49.4000,28.5000) and (49.7000,31.2000) .. (49.9000,33.9000) .. controls (52.3000,35.7000) and (54.6000,37.5000) .. (56.6000,39.2000);

        \path[fill=#2] (36.5000,40.9000) .. controls (36.5000,41.6000) and (37.9000,42.5000) .. (39.4000,42.7000) -- (39.4000,39.2000) .. controls (37.9000,39.4000) and (36.5000,40.2000) .. (36.5000,40.9000);

        \path[fill=#2] (44.9000,40.9000) .. controls (44.9000,41.8000) and (44.0000,42.7000) .. (42.5000,43.2000) .. controls (43.7000,43.6000) and (44.5000,44.4000) .. (44.5000,45.2000) .. controls (44.5000,46.1000) and (43.7000,46.8000) .. (42.4000,47.3000) .. controls (42.2000,47.4000) and (41.4000,47.6000) .. (40.7000,47.6000) .. controls (40.7000,47.6000) and (40.7000,47.6000) .. (40.7000,47.6000) .. controls (40.7000,47.6000) and (40.7000,47.6000) .. (40.7000,47.6000) -- (40.7000,48.7000) .. controls (40.7000,49.0000) and (40.5000,49.3000) .. (40.1000,49.3000) .. controls (39.8000,49.3000) and (39.5000,49.1000) .. (39.5000,48.7000) -- (39.5000,47.6000) .. controls (39.5000,47.6000) and (39.5000,47.6000) .. (39.5000,47.6000) .. controls (39.5000,47.6000) and (39.5000,47.6000) .. (39.5000,47.6000) .. controls (38.8000,47.6000) and (38.0000,47.4000) .. (37.8000,47.3000) .. controls (36.5000,46.9000) and (35.7000,46.1000) .. (35.7000,45.2000) .. controls (35.7000,44.4000) and (36.4000,43.7000) .. (37.7000,43.2000) .. controls (36.0000,42.8000) and (35.0000,41.9000) .. (35.0000,40.9000) .. controls (35.0000,40.0000) and (35.8000,39.2000) .. (37.2000,38.7000) .. controls (35.5000,38.2000) and (34.4000,37.2000) .. (34.4000,36.2000) .. controls (34.4000,35.3000) and (35.1000,34.5000) .. (36.4000,34.0000) .. controls (36.5000,33.7000) and (36.8000,33.4000) .. (37.2000,33.2000) .. controls (37.5000,33.1000) and (37.8000,33.1000) .. (38.1000,33.2000) .. controls (38.4000,33.3000) and (38.6000,33.4000) .. (38.6000,33.7000) .. controls (38.7000,33.9000) and (38.6000,34.2000) .. (38.5000,34.4000) .. controls (38.3000,34.6000) and (38.1000,34.8000) .. (37.8000,34.9000) .. controls (37.6000,35.0000) and (37.4000,35.0000) .. (37.2000,35.0000) .. controls (36.5000,35.4000) and (36.1000,35.9000) .. (36.1000,36.3000) .. controls (36.1000,37.2000) and (37.7000,38.1000) .. (39.4000,38.3000) -- (39.4000,32.0000) .. controls (39.4000,31.6000) and (39.6000,31.4000) .. (39.9000,31.3000) .. controls (40.3000,31.3000) and (40.5000,31.6000) .. (40.5000,32.0000) -- (40.5000,38.3000) .. controls (42.3000,38.1000) and (43.8000,37.2000) .. (43.8000,36.3000) .. controls (43.8000,35.9000) and (43.4000,35.4000) .. (42.7000,35.0000) .. controls (42.5000,35.0000) and (42.3000,35.0000) .. (42.1000,34.9000) .. controls (41.8000,34.8000) and (41.6000,34.6000) .. (41.4000,34.4000) .. controls (41.2000,34.2000) and (41.2000,33.9000) .. (41.2000,33.7000) .. controls (41.3000,33.5000) and (41.5000,33.3000) .. (41.7000,33.2000) .. controls (42.0000,33.1000) and (42.3000,33.1000) .. (42.6000,33.2000) .. controls (43.0000,33.3000) and (43.3000,33.6000) .. (43.4000,34.0000) .. controls (44.6000,34.6000) and (45.4000,35.4000) .. (45.4000,36.2000) .. controls (45.4000,37.3000) and (44.3000,38.2000) .. (42.6000,38.7000) .. controls (44.0000,39.2000) and (44.9000,40.1000) .. (44.9000,40.9000)(44.3000,32.2000) .. controls (42.8000,31.4000) and (41.4000,30.6000) .. (39.9000,29.8000) .. controls (38.3000,30.6000) and (36.7000,31.4000) .. (35.1000,32.4000) .. controls (33.6000,33.3000) and (32.1000,34.2000) .. (30.7000,35.1000) .. controls (30.6000,36.7000) and (30.5000,38.5000) .. (30.5000,40.2000) .. controls (30.5000,41.9000) and (30.5000,43.5000) .. (30.6000,45.1000) .. controls (32.0000,46.0000) and (33.5000,47.0000) .. (35.1000,47.9000) .. controls (36.7000,48.8000) and (38.2000,49.6000) .. (39.8000,50.4000) .. controls (41.2000,49.7000) and (42.7000,48.9000) .. (44.1000,48.1000) .. controls (45.6000,47.2000) and (47.1000,46.3000) .. (48.5000,45.4000) .. controls (48.6000,43.8000) and (48.7000,42.0000) .. (48.7000,40.3000) .. controls (48.7000,38.4000) and (48.6000,36.6000) .. (48.5000,34.9000) .. controls (47.2000,34.0000) and (45.8000,33.1000) .. (44.3000,32.2000);

        \path[fill=#2] (40.5000,42.7000) .. controls (42.0000,42.5000) and (43.4000,41.7000) .. (43.4000,40.9000) .. controls (43.4000,40.2000) and (42.0000,39.3000) .. (40.5000,39.1000) -- (40.5000,42.7000) -- cycle;

        \path[fill=#2] (40.5000,46.8000) .. controls (40.5000,46.8000) and (40.5000,46.8000) .. (40.5000,46.8000) .. controls (41.7000,46.6000) and (43.0000,45.9000) .. (43.0000,45.2000) .. controls (43.0000,44.5000) and (41.7000,43.8000) .. (40.5000,43.6000) -- (40.5000,46.8000) -- cycle;

        \path[fill=#2] (6.0000,26.7000) .. controls (6.3000,26.8000) and (6.6000,26.7000) .. (6.7000,26.4000) .. controls (6.8000,26.1000) and (6.7000,25.8000) .. (6.4000,25.8000) .. controls (6.2000,25.7000) and (5.9000,25.8000) .. (5.8000,26.1000) .. controls (5.7000,26.3000) and (5.8000,26.6000) .. (6.0000,26.7000)(3.5000,33.8000) .. controls (5.2000,34.3000) and (6.7000,33.6000) .. (7.2000,31.8000) .. controls (7.7000,30.1000) and (6.7000,28.7000) .. (5.0000,28.3000) .. controls (3.3000,27.8000) and (1.8000,28.5000) .. (1.3000,30.3000) .. controls (0.8000,32.0000) and (1.8000,33.4000) .. (3.5000,33.8000)(3.7000,33.0000) .. controls (2.5000,32.7000) and (1.6000,31.8000) .. (2.0000,30.5000) .. controls (2.3000,29.2000) and (3.6000,28.9000) .. (4.7000,29.2000) .. controls (5.8000,29.5000) and (6.8000,30.4000) .. (6.4000,31.7000) .. controls (6.1000,33.0000) and (4.8000,33.3000) .. (3.7000,33.0000)(0.8000,39.2000) -- (0.9000,38.2000) .. controls (1.0000,37.0000) and (1.8000,36.1000) .. (3.2000,36.2000) .. controls (4.7000,36.3000) and (5.4000,37.1000) .. (5.3000,38.6000) -- (5.2000,39.5000) -- (0.8000,39.2000) -- cycle(5.9000,40.3000) -- (6.0000,38.6000) .. controls (6.1000,36.7000) and (5.2000,35.4000) .. (3.2000,35.3000) .. controls (1.4000,35.2000) and (0.2000,36.5000) .. (0.1000,38.3000) -- (0.0000,40.0000) -- (5.9000,40.3000) -- cycle(6.6000,46.9000) -- (6.5000,46.1000) -- (1.7000,46.8000) -- (1.7000,46.8000) -- (6.1000,43.5000) -- (6.0000,42.4000) -- (0.1000,43.2000) -- (0.2000,44.0000) -- (4.9000,43.3000) -- (4.9000,43.3000) -- (0.5000,46.5000) -- (0.7000,47.6000) -- (6.6000,46.9000) -- cycle(5.3000,50.6000) -- (6.0000,52.9000) -- (2.8000,52.7000) -- (5.3000,50.6000) -- cycle(8.7000,53.8000) -- (8.4000,53.0000) -- (6.8000,52.9000) -- (5.9000,50.2000) -- (7.1000,49.1000) -- (6.8000,48.2000) -- (2.0000,52.5000) -- (2.3000,53.3000) -- (8.7000,53.8000) -- cycle(6.1000,59.9000) -- (5.7000,59.2000) .. controls (5.3000,58.5000) and (5.4000,58.0000) .. (5.9000,57.7000) .. controls (6.4000,57.4000) and (6.9000,57.7000) .. (7.3000,58.4000) -- (7.6000,59.0000) -- (6.1000,59.9000) -- cycle(11.1000,58.2000) -- (10.7000,57.5000) -- (8.4000,58.8000) -- (8.2000,58.4000) .. controls (8.0000,58.0000) and (7.9000,57.8000) .. (8.2000,57.3000) -- (9.6000,55.5000) -- (9.1000,54.7000) -- (7.6000,56.8000) .. controls (7.4000,57.1000) and (7.3000,57.3000) .. (7.4000,57.6000) -- (7.4000,57.6000) .. controls (7.0000,57.1000) and (6.3000,56.8000) .. (5.6000,57.2000) .. controls (4.4000,57.9000) and (4.7000,59.0000) .. (5.3000,60.0000) -- (5.9000,61.0000) -- (11.1000,58.2000) -- cycle(7.4000,63.5000) -- (8.0000,64.1000) -- (10.6000,62.0000) .. controls (11.5000,61.3000) and (12.3000,61.1000) .. (13.0000,61.9000) .. controls (13.7000,62.7000) and (13.3000,63.5000) .. (12.4000,64.2000) -- (9.8000,66.3000) -- (10.3000,67.0000) -- (13.0000,64.8000) .. controls (14.3000,63.7000) and (14.4000,62.5000) .. (13.5000,61.4000) .. controls (12.6000,60.3000) and (11.4000,60.2000) .. (10.1000,61.3000) -- (7.4000,63.5000) -- cycle(12.2000,68.7000) .. controls (12.4000,69.1000) and (12.7000,69.5000) .. (13.0000,69.8000) .. controls (14.4000,71.0000) and (16.2000,70.9000) .. (17.3000,69.5000) .. controls (18.5000,68.1000) and (18.4000,66.3000) .. (16.9000,65.1000) .. controls (16.6000,64.9000) and (16.2000,64.5000) .. (15.7000,64.4000) -- (15.3000,65.0000) .. controls (15.6000,65.1000) and (16.1000,65.3000) .. (16.5000,65.6000) .. controls (17.4000,66.4000) and (17.6000,67.8000) .. (16.7000,68.8000) .. controls (15.8000,69.8000) and (14.5000,70.0000) .. (13.5000,69.1000) .. controls (13.2000,68.8000) and (12.9000,68.5000) .. (12.8000,68.1000) -- (12.2000,68.7000) -- cycle(20.0000,70.8000) .. controls (20.1000,70.6000) and (20.1000,70.2000) .. (19.9000,70.1000) .. controls (19.7000,69.9000) and (19.4000,70.0000) .. (19.2000,70.3000) .. controls (19.0000,70.5000) and (19.1000,70.8000) .. (19.3000,71.0000) .. controls (19.5000,71.0000) and (19.9000,71.0000) .. (20.0000,70.8000)(26.6000,74.6000) .. controls (27.4000,73.0000) and (27.0000,71.4000) .. (25.3000,70.6000) .. controls (23.7000,69.8000) and (22.2000,70.5000) .. (21.4000,72.2000) .. controls (20.7000,73.8000) and (21.1000,75.4000) .. (22.7000,76.2000) .. controls (24.4000,76.9000) and (25.9000,76.2000) .. (26.6000,74.6000)(25.8000,74.2000) .. controls (25.3000,75.3000) and (24.2000,76.0000) .. (23.1000,75.5000) .. controls (21.9000,75.0000) and (21.8000,73.7000) .. (22.3000,72.6000) .. controls (22.8000,71.5000) and (23.8000,70.8000) .. (25.0000,71.3000) .. controls (26.3000,71.9000) and (26.3000,73.2000) .. (25.8000,74.2000)(31.2000,78.2000) -- (30.2000,77.9000) .. controls (29.0000,77.6000) and (28.3000,76.6000) .. (28.7000,75.2000) .. controls (29.1000,73.7000) and (30.1000,73.3000) .. (31.5000,73.6000) -- (32.4000,73.8000) -- (31.2000,78.2000) -- cycle(33.4000,73.4000) -- (31.7000,73.0000) .. controls (29.8000,72.5000) and (28.4000,73.2000) .. (27.9000,75.1000) .. controls (27.4000,76.9000) and (28.5000,78.3000) .. (30.2000,78.7000) -- (31.9000,79.2000) -- (33.4000,73.4000) -- cycle(39.9000,74.0000) -- (39.0000,74.0000) -- (38.7000,78.9000) -- (38.7000,78.9000) -- (36.3000,73.9000) -- (35.2000,73.8000) -- (34.9000,79.7000) -- (35.7000,79.7000) -- (36.0000,74.9000) -- (36.0000,74.9000) -- (38.3000,79.8000) -- (39.5000,79.9000) -- (39.9000,74.0000) -- cycle(44.9000,73.6000) -- (41.6000,74.0000) -- (41.7000,74.7000) -- (44.2000,74.4000) -- (44.4000,76.3000) -- (42.2000,76.6000) -- (42.3000,77.3000) -- (44.5000,77.0000) -- (44.7000,78.7000) -- (42.2000,79.0000) -- (42.3000,79.7000) -- (45.6000,79.3000) -- (44.9000,73.6000) -- cycle(48.0000,78.9000) .. controls (48.5000,79.0000) and (48.9000,78.9000) .. (49.3000,78.8000) .. controls (51.0000,78.3000) and (52.0000,76.8000) .. (51.5000,75.0000) .. controls (51.0000,73.2000) and (49.4000,72.4000) .. (47.6000,72.9000) .. controls (47.3000,73.0000) and (46.7000,73.2000) .. (46.4000,73.5000) -- (46.7000,74.3000) .. controls (46.9000,74.0000) and (47.4000,73.8000) .. (47.9000,73.7000) .. controls (49.1000,73.4000) and (50.3000,74.0000) .. (50.7000,75.3000) .. controls (51.1000,76.6000) and (50.5000,77.8000) .. (49.2000,78.2000) .. controls (48.8000,78.3000) and (48.4000,78.3000) .. (47.9000,78.2000) -- (48.0000,78.9000) -- cycle(57.8000,72.2000) .. controls (57.0000,70.6000) and (55.5000,69.9000) .. (53.9000,70.7000) .. controls (52.3000,71.5000) and (51.9000,73.1000) .. (52.7000,74.7000) .. controls (53.5000,76.3000) and (55.0000,77.0000) .. (56.6000,76.2000) .. controls (58.2000,75.4000) and (58.5000,73.7000) .. (57.8000,72.2000)(57.0000,72.6000) .. controls (57.5000,73.7000) and (57.4000,75.0000) .. (56.3000,75.5000) .. controls (55.1000,76.1000) and (54.1000,75.3000) .. (53.5000,74.3000) .. controls (53.0000,73.2000) and (53.0000,72.0000) .. (54.2000,71.4000) .. controls (55.4000,70.8000) and (56.4000,71.5000) .. (57.0000,72.6000)(63.5000,71.0000) -- (62.7000,71.6000) .. controls (61.7000,72.3000) and (60.5000,72.2000) .. (59.7000,71.1000) .. controls (58.8000,69.9000) and (59.1000,68.8000) .. (60.3000,68.0000) -- (61.0000,67.5000) -- (63.5000,71.0000) -- cycle(61.2000,66.2000) -- (59.8000,67.2000) .. controls (58.2000,68.3000) and (57.8000,69.9000) .. (59.0000,71.5000) .. controls (60.1000,73.0000) and (61.8000,73.1000) .. (63.3000,72.0000) -- (64.7000,71.0000) -- (61.2000,66.2000) -- cycle(65.0000,66.2000) .. controls (64.8000,66.0000) and (64.5000,66.0000) .. (64.3000,66.2000) .. controls (64.1000,66.4000) and (64.1000,66.7000) .. (64.3000,66.9000) .. controls (64.5000,67.1000) and (64.8000,67.1000) .. (65.0000,66.9000) .. controls (65.1000,66.7000) and (65.2000,66.4000) .. (65.0000,66.2000)(70.2000,60.6000) .. controls (68.8000,59.5000) and (67.1000,59.5000) .. (66.0000,61.0000) .. controls (64.9000,62.4000) and (65.2000,64.0000) .. (66.7000,65.1000) .. controls (68.1000,66.2000) and (69.8000,66.2000) .. (70.9000,64.7000) .. controls (72.0000,63.3000) and (71.6000,61.7000) .. (70.2000,60.6000)(69.7000,61.3000) .. controls (70.6000,62.0000) and (71.1000,63.2000) .. (70.3000,64.3000) .. controls (69.5000,65.3000) and (68.2000,65.2000) .. (67.3000,64.4000) .. controls (66.4000,63.7000) and (65.8000,62.5000) .. (66.7000,61.4000) .. controls (67.5000,60.3000) and (68.7000,60.6000) .. (69.7000,61.3000)(74.8000,56.8000) -- (74.3000,57.7000) .. controls (73.7000,58.8000) and (72.6000,59.2000) .. (71.3000,58.6000) .. controls (70.0000,57.9000) and (69.7000,56.8000) .. (70.4000,55.6000) -- (70.8000,54.8000) -- (74.8000,56.8000) -- cycle(70.6000,53.7000) -- (69.8000,55.2000) .. controls (68.9000,56.9000) and (69.2000,58.5000) .. (71.0000,59.4000) .. controls (72.6000,60.3000) and (74.2000,59.5000) .. (75.0000,58.0000) -- (75.8000,56.5000) -- (70.6000,53.7000) -- cycle(72.6000,47.5000) -- (72.4000,48.3000) -- (77.1000,49.7000) -- (77.1000,49.7000) -- (71.7000,50.9000) -- (71.4000,51.9000) -- (77.1000,53.6000) -- (77.3000,52.8000) -- (72.7000,51.5000) -- (72.7000,51.5000) -- (78.0000,50.3000) -- (78.3000,49.2000) -- (72.6000,47.5000) -- cycle(73.2000,42.4000) -- (72.8000,45.7000) -- (73.5000,45.8000) -- (73.8000,43.3000) -- (75.7000,43.5000) -- (75.5000,45.7000) -- (76.2000,45.8000) -- (76.4000,43.6000) -- (78.1000,43.8000) -- (78.0000,46.3000) -- (78.7000,46.4000) -- (79.1000,43.1000) -- (73.2000,42.4000) -- cycle(73.2000,39.3000) -- (73.2000,40.1000) -- (79.1000,40.0000) -- (79.1000,39.2000) -- (73.2000,39.3000) -- cycle(78.9000,36.9000) .. controls (79.0000,36.5000) and (79.0000,36.0000) .. (79.0000,35.5000) .. controls (78.8000,33.7000) and (77.4000,32.6000) .. (75.6000,32.8000) .. controls (73.7000,33.1000) and (72.7000,34.5000) .. (73.0000,36.3000) .. controls (73.0000,36.6000) and (73.1000,37.2000) .. (73.4000,37.6000) -- (74.2000,37.4000) .. controls (74.0000,37.1000) and (73.8000,36.6000) .. (73.7000,36.2000) .. controls (73.5000,35.0000) and (74.3000,33.9000) .. (75.7000,33.7000) .. controls (77.0000,33.5000) and (78.1000,34.3000) .. (78.3000,35.6000) .. controls (78.4000,36.0000) and (78.3000,36.5000) .. (78.1000,36.9000) -- (78.9000,36.9000) -- cycle(77.8000,30.3000) .. controls (77.8000,29.9000) and (77.8000,29.4000) .. (77.6000,29.0000) .. controls (77.3000,28.0000) and (76.5000,27.3000) .. (75.5000,27.6000) .. controls (73.7000,28.1000) and (74.8000,30.5000) .. (73.5000,30.8000) .. controls (72.8000,31.0000) and (72.4000,30.4000) .. (72.2000,30.0000) .. controls (72.1000,29.6000) and (72.1000,29.1000) .. (72.1000,28.7000) -- (71.3000,28.8000) .. controls (71.3000,29.2000) and (71.3000,29.6000) .. (71.4000,30.0000) .. controls (71.7000,31.1000) and (72.5000,31.9000) .. (73.7000,31.6000) .. controls (75.6000,31.1000) and (74.6000,28.7000) .. (75.7000,28.4000) .. controls (76.3000,28.2000) and (76.6000,28.7000) .. (76.7000,29.2000) .. controls (76.8000,29.6000) and (76.8000,29.9000) .. (76.8000,30.3000) -- (77.8000,30.3000) -- cycle(72.7000,25.6000) .. controls (72.5000,25.7000) and (72.3000,26.0000) .. (72.4000,26.3000) .. controls (72.5000,26.5000) and (72.8000,26.7000) .. (73.1000,26.6000) .. controls (73.4000,26.5000) and (73.5000,26.2000) .. (73.4000,26.0000) .. controls (73.3000,25.6000) and (73.0000,25.5000) .. (72.7000,25.6000)(66.2000,19.4000) -- (66.7000,20.1000) -- (71.0000,17.3000) -- (71.0000,17.3000) -- (67.7000,21.7000) -- (68.1000,22.4000) -- (73.5000,21.3000) -- (73.5000,21.3000) -- (69.2000,24.1000) -- (69.7000,24.8000) -- (74.7000,21.6000) -- (73.9000,20.4000) -- (68.8000,21.5000) -- (71.9000,17.3000) -- (71.1000,16.1000) -- (66.2000,19.4000) -- cycle(63.2000,16.3000) -- (65.5000,18.7000) -- (66.0000,18.2000) -- (64.3000,16.4000) -- (68.0000,12.8000) -- (67.4000,12.2000) -- (63.2000,16.3000) -- cycle(65.4000,10.3000) -- (64.7000,9.8000) -- (62.6000,12.4000) .. controls (61.9000,13.3000) and (61.1000,13.7000) .. (60.3000,13.0000) .. controls (59.4000,12.3000) and (59.6000,11.5000) .. (60.3000,10.6000) -- (62.5000,8.0000) -- (61.8000,7.5000) -- (59.7000,10.2000) .. controls (58.7000,11.5000) and (58.8000,12.7000) .. (59.9000,13.6000) .. controls (61.1000,14.5000) and (62.3000,14.3000) .. (63.3000,13.0000) -- (65.4000,10.3000) -- cycle(53.3000,9.6000) -- (54.1000,10.0000) -- (56.3000,5.3000) -- (57.9000,6.0000) -- (58.2000,5.3000) -- (54.1000,3.4000) -- (53.8000,4.1000) -- (55.5000,4.9000) -- (53.3000,9.6000) -- cycle(51.1000,1.9000) -- (51.9000,2.1000) -- (52.1000,1.3000) -- (51.3000,1.1000) -- (51.1000,1.9000) -- cycle(49.7000,1.5000) -- (50.5000,1.7000) -- (50.7000,0.9000) -- (50.0000,0.7000) -- (49.7000,1.5000) -- cycle(50.7000,6.2000) -- (48.4000,5.5000) -- (50.4000,3.0000) -- (50.7000,6.2000) -- cycle(46.1000,7.1000) -- (47.0000,7.4000) -- (48.0000,6.1000) -- (50.8000,7.0000) -- (51.0000,8.6000) -- (51.9000,8.9000) -- (51.2000,2.5000) -- (50.3000,2.2000) -- (46.1000,7.1000) -- cycle(43.6000,6.9000) -- (44.4000,7.0000) -- (45.1000,1.9000) -- (46.9000,2.1000) -- (47.0000,1.4000) -- (42.5000,0.8000) -- (42.4000,1.5000) -- (44.2000,1.7000) -- (43.6000,6.9000) -- cycle(40.0000,6.6000) -- (40.8000,6.6000) -- (41.0000,0.7000) -- (40.2000,0.7000) -- (40.0000,6.6000) -- cycle(37.6000,0.9000) .. controls (37.2000,0.8000) and (36.7000,0.8000) .. (36.3000,0.8000) .. controls (35.2000,0.9000) and (34.5000,1.5000) .. (34.6000,2.6000) .. controls (34.8000,4.4000) and (37.3000,3.8000) .. (37.4000,5.2000) .. controls (37.5000,5.9000) and (36.8000,6.2000) .. (36.3000,6.3000) .. controls (35.9000,6.3000) and (35.3000,6.2000) .. (35.1000,6.1000) -- (35.1000,6.9000) .. controls (35.5000,7.0000) and (35.9000,7.1000) .. (36.3000,7.0000) .. controls (37.5000,6.9000) and (38.4000,6.3000) .. (38.3000,5.1000) .. controls (38.1000,3.2000) and (35.6000,3.6000) .. (35.5000,2.5000) .. controls (35.4000,1.9000) and (36.0000,1.7000) .. (36.5000,1.6000) .. controls (36.9000,1.6000) and (37.2000,1.6000) .. (37.6000,1.7000) -- (37.6000,0.9000) -- cycle(29.4000,2.8000) -- (30.2000,2.6000) .. controls (31.0000,2.4000) and (31.4000,2.7000) .. (31.6000,3.2000) .. controls (31.7000,3.8000) and (31.4000,4.2000) .. (30.6000,4.4000) -- (29.9000,4.6000) -- (29.4000,2.8000) -- cycle(29.8000,8.0000) -- (30.6000,7.8000) -- (30.0000,5.3000) -- (30.5000,5.2000) .. controls (31.0000,5.1000) and (31.2000,5.0000) .. (31.6000,5.5000) -- (33.0000,7.3000) -- (33.9000,7.1000) -- (32.3000,5.1000) .. controls (32.0000,4.8000) and (31.9000,4.7000) .. (31.6000,4.7000) -- (31.6000,4.7000) .. controls (32.2000,4.4000) and (32.6000,3.8000) .. (32.5000,3.1000) .. controls (32.2000,1.7000) and (31.0000,1.8000) .. (29.9000,2.1000) -- (28.5000,2.4000) -- (29.8000,8.0000) -- cycle(25.1000,9.9000) -- (28.1000,8.6000) -- (27.8000,7.9000) -- (25.5000,8.9000) -- (24.7000,7.1000) -- (26.8000,6.2000) -- (26.5000,5.5000) -- (24.4000,6.4000) -- (23.7000,4.8000) -- (26.0000,3.8000) -- (25.7000,3.1000) -- (22.7000,4.4000) -- (25.1000,9.9000) -- cycle(20.7000,5.4000) -- (20.0000,5.9000) -- (21.2000,11.3000) -- (21.2000,11.3000) -- (16.9000,7.9000) -- (16.1000,8.4000) -- (21.2000,12.2000) -- (22.1000,11.6000) -- (20.7000,5.4000) -- cycle(17.7000,14.8000) -- (18.4000,14.3000) -- (14.6000,9.7000) -- (13.9000,10.2000) -- (17.7000,14.8000) -- cycle(13.0000,19.6000) -- (13.6000,19.0000) -- (10.0000,15.7000) -- (10.0000,15.7000) -- (15.4000,17.0000) -- (16.1000,16.2000) -- (11.7000,12.2000) -- (11.1000,12.8000) -- (14.6000,16.0000) -- (14.6000,16.0000) -- (9.3000,14.7000) -- (8.5000,15.6000) -- (13.0000,19.6000) -- cycle(6.9000,18.0000) -- (6.5000,18.7000) -- (9.4000,20.4000) .. controls (10.4000,21.0000) and (10.9000,21.7000) .. (10.3000,22.6000) .. controls (9.7000,23.5000) and (8.9000,23.4000) .. (7.9000,22.9000) -- (5.0000,21.2000) -- (4.6000,21.9000) -- (7.6000,23.7000) .. controls (9.0000,24.6000) and (10.2000,24.3000) .. (11.0000,23.0000) .. controls (11.7000,21.7000) and (11.4000,20.6000) .. (10.0000,19.7000) -- (6.9000,18.0000) -- cycle;
    \end{scope}
    \end{tikzpicture}
}
% }}}

\newcommand\vslogo[2]{%{{{
\begin{tikzpicture}[y=0.80pt, x=0.8pt,yscale=-1, inner sep=0pt, outer sep=0pt]
  \pgftransformscale{#2/30mm}
\begin{scope}[cm={{1.25,0.0,0.0,-1.25,(0.0,177.1675)}}]
      \begin{scope}[shift={(99.2129,113.3867)}]
        \path[fill=#1,nonzero rule] (0.0000,0.0000) -- (-28.3470,0.0000) ..
          controls (-28.3470,-28.3470) and (0.0000,-85.0390) .. (0.0000,-85.0390) --
          (28.3470,-85.0390) .. controls (56.6930,-33.0700) and (-15.0650,-50.8970) ..
          (0.0000,0.0000);
      \end{scope}
      \begin{scope}[shift={(155.9053,113.3867)}]
        \path[fill=#1,nonzero rule] (0.0000,0.0000) -- (-28.3460,0.0000) ..
          controls (-46.8860,-48.1530) and (27.7810,-30.4040) .. (0.0010,-85.0390) --
          (14.1740,-85.0390) -- (14.1740,-70.8660) -- (14.1740,-14.1730) .. controls
          (14.1740,-6.3460) and (7.8280,0.0000) .. (0.0000,0.0000);
      \end{scope}
      \begin{scope}[shift={(28.3467,99.2139)}]
        \path[fill=#1,nonzero rule] (0.0000,0.0000) -- (0.0000,-56.6930) ..
          controls (0.0000,-64.5210) and (6.3460,-70.8660) .. (14.1730,-70.8660) --
          (42.5200,-70.8660) .. controls (42.5200,-70.8660) and (14.1730,-13.9800) ..
          (14.1730,14.1730) .. controls (6.3460,14.1730) and (0.0000,7.8270) ..
          (0.0000,0.0000);
      \end{scope}
      \begin{scope}[shift={(99.4272,113.3867)}]
        \path[fill=white,nonzero rule] (0.0000,0.0000) .. controls (-15.0650,-50.8970)
          and (56.6930,-33.0700) .. (28.3470,-85.0390) -- (56.6930,-85.0390) .. controls
          (84.4740,-30.4040) and (9.8070,-48.1530) .. (28.3470,0.0000) --
          (0.0000,0.0000) -- cycle;
      \end{scope}
          \shade[shading=axis,bottom color=#1!10,top color=#1,shading angle=-30] (85.0390,60.2370) .. controls (92.1260,42.5210) and
            (99.2130,28.3460) .. (99.2130,28.3460) -- (99.2130,28.3460) .. controls
            (99.2130,28.3460) and (109.5400,49.0740) .. (117.7260,71.6970) --
            (117.7260,71.6970) .. controls (108.3930,77.9380) and (98.3750,84.2770) ..
            (97.2280,97.7960) -- (97.2280,97.7960) .. controls (94.6300,85.8490) and
            (89.8380,72.2330) .. (85.0390,60.2370);
      \begin{scope}[shift={(42.5195,113.3867)}]
        \path[fill=white,nonzero rule] (0.0000,0.0000) .. controls (0.0000,-28.1530)
          and (28.3470,-85.0390) .. (28.3470,-85.0390) -- (56.6930,-85.0390) .. controls
          (56.6930,-85.0390) and (28.3470,-28.3470) .. (28.3470,0.0000) --
          (0.0000,0.0000) -- cycle;
      \end{scope}
\end{scope}
\end{tikzpicture}
}
%}}}

\newcommand{\ccurl}{%
    \expandafter\ifx\@license\cczero\deen{\url{https://creativecommons.org/publicdomain/zero/1.0/deed.de}}{\url{https://creativecommons.org/publicdomain/zero/1.0/}}\else%
    \expandafter\ifx\@license\ccbysa\deen{\url{https://creativecommons.org/licenses/by-sa/4.0/deed.de}}{\url{https://creativecommons.org/licenses/by-sa/4.0/}}\else%
    \expandafter\ifx\@license\ccbynd\deen{\url{https://creativecommons.org/licenses/by-nd/4.0/deed.de}}{\url{https://creativecommons.org/licenses/by-nd/4.0/}}\else%
    \expandafter\ifx\@license\ccbync\deen{\url{https://creativecommons.org/licenses/by-nc/4.0/deed.de}}{\url{https://creativecommons.org/licenses/by-nc/4.0/}}\else%
    \expandafter\ifx\@license\ccbyncsa\deen{\url{https://creativecommons.org/licenses/by-nc-sa/4.0/deed.de}}{\url{https://creativecommons.org/licenses/by-nc-sa/4.0/}}\else%
    \expandafter\ifx\@license\ccbyncnd\deen{\url{https://creativecommons.org/licenses/by-nc-nd/4.0/deed.de}}{\url{https://creativecommons.org/licenses/by-nc-nd/4.0/}}\else%
    \expandafter\ifx\@license\ccby\deen{\url{https://creativecommons.org/licenses/by/4.0/deed.de}}{\url{https://creativecommons.org/licenses/by/4.0/}}\else%
    \colorbox{red}{Error: Invalid license!}\fi\fi\fi\fi\fi\fi\fi}

\renewcommand{\maketitle}{
    {\begingroup
            \thispagestyle{empty}
            \begin{tikzpicture}[remember picture, overlay]
                \draw[fill=uulm-akzent!20,draw=uulm-akzent!20] ($ (current page.south west) + (0, 17.7cm) $) rectangle ++(\paperwidth, 6.8cm);
                \draw[fill=uulm-in,draw=uulm-in] ($ (current page.south west) + (0, 14.6cm) $) rectangle ++(\paperwidth, 3.1cm);

                \@titlegraphic

                \node[inner sep=0, anchor=south east, align=flush right, text width=17cm, minimum height=3.1cm] at ($ (current page.south west) + (19cm, 14.6cm) $) {%
                    {\sffamily\color{white}\huge\baselineskip=25pt\@title\par}};

                \node[inner sep=0, anchor=north east, align=flush right, text width=17cm] at ($ (current page.south west) + (19cm, 12.8cm) $) {{\sffamily
                        {\color{uulm-akzent} \@authorprefix}%
                        \ifthenelse{\boolean{lesscolor}}{%
                            {\color{uulm-in} \LARGE\@author\\}%
                        }{%
                            {\color{uulm} \LARGE\@author\\}%
                        }%
                        {\color{uulm-akzent} \@studentnumber\\}
                        \vspace{.5cm}
                        \ifthenelse{\equal{\@thesistype}{}}{}{\color{uulm-in} \@thesistype\\}
                        \ifthenelse{\equal{\@internalid}{}}{}{
                            \ifthenelse{\equal{\@internalid}{VS-2019-B18} \or \equal{\@internalid}{VS-TODO-XYZ}}{
                                \ClassWarningNoLine{vs-thesis}{Default internalid provided. Ask your supervisor for an id for your report.}
                                \colorbox{red}{
                                    \begin{tikzpicture}[y=1.5em, x=1.5em, inner sep=0pt, outer sep=0pt]
                                         \path[fill=white] (0.344, -0.3704)
                                            -- (0.291, -0.3704)
                                            -- (0.291, -0.2381)
                                            -- (0.344, -0.2381)

                                            (0.344, -0.4763)
                                            -- (0.291, -0.4763)
                                            -- (0.291, -0.4233)
                                            -- (0.344, -0.4233)

                                            (0.0265, -0.5556)
                                            -- (0.6085, -0.5556)
                                            -- (0.3175, -0.0529)
                                            -- (0.0265, -0.5556)
                                            -- cycle;
                                     \end{tikzpicture}
                                    \color{white} \@internalid
                                }\\
                            }{\color{uulm-akzent} \@internalid\\}
                        }
                        \ifthenelse{\equal{\@internalid\@thesistype}{}}{}{\vspace{.5cm}}
                        \ifthenelse{\equal{\@examiners}{}}{}{
                            {\color{uulm-in} \deen{Begutachtung}{Examined by}\\}
                            {\color{uulm-akzent} \@examiners\\}
                            \vspace{.5cm}
                        }
                        \ifthenelse{\equal{\@supervisors}{}}{}{
                            {\color{uulm-in} \deen{Betreuung}{Supervised by}\\}
                            {\color{uulm-akzent} \@supervisors\\}
                            \vspace{.5cm}
                        }
                        {\color{uulm-akzent} \@institution\\}
                        \vspace{.5cm}
                        {\color{uulm-akzent} \@submissiondate}
                    }};

                \node[inner sep=0, anchor=north east, align=flush right, text width=15cm] at ($ (current page.south west) + (19cm, 2.8cm) $) {\color{uulm-akzent}\Large\expandafter\ifx\@license\ccCopy~\else\@license\fi};

                \node[inner sep=0, anchor=north east] at ($ (current page.south west) + (19cm, 28cm) $) {\uulmlogo{white}{black}{85mm}};
                \node[inner sep=0, anchor=north west] at ($ (current page.south west) + (2cm, 27.9cm) $) {\vslogo{uulm-in}{15mm}};
            \end{tikzpicture}
        \endgroup}

    {\begingroup%
            \pagebreak%
            \thispagestyle{empty}%
            \footnotesize%
            \@attributions%
            \par\vspace*{\fill}%
            \noindent \textcopyright\ \the\year\ \@author
            \begin{tabbing}%
                \hspace*{6em} \= \kill
                \deen{Fassung vom}{Issued}: \> \today\\
                \\
                \parbox{6em}{\normalsize\expandafter\ifx\@license\ccCopy~\else\@license\fi}
                \> \expandafter\ifx\@license\ccCopy~\else\parbox[t]{0.85\textwidth}{\raggedright
                    \deen{Dieses Werk ist unter einer Creative Commons Lizenz vom Typ}{This work is licensed under a Creative Commons}
                    \expandafter\ifx\@license\cczero\deen{Kein Urheberrechtsschutz}{Public Domain}\else%
                    \expandafter\ifx\@license\ccbysa\deen{Namensnennung - Weitergabe unter gleichen Bedingungen}{Attribution-ShareAlike}\else%
                    \expandafter\ifx\@license\ccbynd\deen{Namensnennung - Keine Bearbeitung}{Attribution-NoDerivatives}\else%
                    \expandafter\ifx\@license\ccbync\deen{Namensnennung - Nicht kommerziell}{Attribution-NonCommercial}\else%
                    \expandafter\ifx\@license\ccbyncsa\deen{Namensnennung - Nicht-kommerziell - Weitergabe unter gleichen Bedingungen}{Attribution-NonCommercial-ShareAlike}\else%
                    \expandafter\ifx\@license\ccbyncnd\deen{Namensnennung - Nicht-kommerziell - Keine Bearbeitung}{Attribution-NonCommercial-NoDerivatives}\else%
                    \expandafter\ifx\@license\ccby\deen{Namensnennung}{Attribution}\else\colorbox{red}{Error: Invalid license!}
                    \fi\fi\fi\fi\fi\fi\fi
                    \deen{ zugänglich}{ License}.\\[.5em]
                    \deen{Um eine Kopie dieser Lizenz einzusehen, konsultieren Sie}{To view a copy of this license, visit} \ccurl
                    \deen{ oder wenden Sie sich brieflich an}{ or send a letter to} Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.}\fi\\
            \end{tabbing}
        \endgroup}

    \@independencedeclaration
}

\makeatletter
%
% PAGE LAYOUT
%
\settypeblocksize{23.45cm}{11.85cm}{*}
\setlrmargins{3.141592653589793cm}{*}{*}
\setulmargins{*}{*}{*}
\setmarginnotes{0.8cm}{3.5cm}{1cm}
\checkandfixthelayout

%
% CODE LISTINGS
%
\RequirePackage{listings}

\lstdefinelanguage{JavaScript}{
  keywords={typeof, new, true, false, catch, function, return, null, catch, switch, var, if, in, while, do, else, case, break, const},
  ndkeywords={class, export, boolean, throw, implements, import, this},
  identifierstyle=\color{uulm}\ttfamily,
  keywordstyle=\color{uulm-med}\ttfamily,
  basicstyle=\footnotesize\ttfamily,
  commentstyle=\color{uulm-akzent}\ttfamily,
  stringstyle=\color{uulm-mawi}\ttfamily,
  numbers=left,
  numberstyle=\color{gray!80}\scriptsize,
  stepnumber=1,
  numbersep=8pt,
  showstringspaces=false,
  breaklines=true,
  belowcaptionskip=.75\baselineskip,
  comment=[l]{//},
  morecomment=[s]{/*}{*/},
  morestring=[b]',
  morestring=[b]"
}

% set defaults
\lstset{
    language=JavaScript,
    captionpos=b,
    frame=tb
}

% Bibliography.
\RequirePackage[autolang=hyphen, backend=biber]{biblatex}

% Algorithms.
% Use default Memoir floats for algorithms
\newcommand{\algorithmname}{Algorithm}
\newcommand{\listalgorithmname}{List of Algorithms}
\newlistof{listofalgorithms}{loa}{\listalgorithmname}
\newfloat[chapter]{algorithm}{loa}{\algorithmname}
\newfixedcaption{\falgcaption}{algorithm}
\newlistentry[chapter]{algorithm}{loa}{0}
\cftsetindents{algorithm}{0em}{2.3em}
\makeatletter
\g@addto@macro\insertchapterspace{\addtocontents{loa}{\protect\addvspace{10pt}}}
\makeatother

\RequirePackage{algpseudocode} % Algorithms.

\RequirePackage[pdfencoding=auto,pdfborderstyle={/S/U/W 1},linkbordercolor=ref,colorlinks,linkcolor=ref,citecolor=cite,urlcolor=url]{hyperref} % Clickable references.

\RequirePackage[all]{hypcap} % Anchors links to the beginning of their respective floats. Load after hyperref.

\RequirePackage{booktabs}
\RequirePackage{multirow}

\RequirePackage{pgfplots}
\pgfplotsset{compat=1.8}
\pgfplotscreateplotcyclelist{uulm}{%
  uulm,semithick,every mark/.append style={fill=uulm!60},mark=*\\%
  uulm-mawi,semithick,every mark/.append style={fill=uulm-mawi!60},mark=square*\\%
  uulm-nawi,semithick,every mark/.append style={fill=uulm-nawi!60},mark=otimes*\\%
  uulm-med,semithick,every mark/.append style={fill=uulm-med!60},mark=star\\%
  uulm-akzent,semithick,every mark/.append style={fill=uulm-akzent!60},mark=diamond*\\%
  uulm-in,semithick,every mark/.append style={solid,fill=uulm-in!60},mark=*\\%
}
\pgfplotsset{cycle list name=uulm}

% Contains some macros which are commonly used in a thesis.

% Abbreviations
\newcommand*{\eg}{e.\,g.,\@\@\xspace}
\newcommand*{\ie}{i.\,e.,\@\@\xspace}
\newcommand*{\cf}{cf.,\@\xspace}
\newcommand*{\Eg}{E.\,g.,\@\xspace}
\newcommand*{\Ie}{I.\,e.,\@\xspace}
\newcommand*{\Cf}{Cf.,\@\xspace}
\newcommand*{\etal}{et.\@ al.\@\xspace}
\makeatletter
\newcommand*{\etc}{\@ifnextchar{.}{etc}{etc.\@\xspace}}
\makeatother
