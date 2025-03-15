%---------------------------------------------------------------------------%
%-                                                                         -%
%-                           Document Class                                -%
%-                                                                         -%
%---------------------------------------------------------------------------%
%- Copyright (C) Huangrui Mo <huangrui.mo@gmail.com>
%- This is free software: you can redistribute it and/or modify it
%- under the terms of the GNU General Public License as published by
%- the Free Software Foundation, either version 3 of the License, or
%- (at your option) any later version.
%---------------------------------------------------------------------------%
%->> Identification
%---------------------------------------------------------------------------%
\NeedsTeXFormat{LaTeX2e}%
\newcommand*\@iflatexlater{\@ifl@t@r\fmtversion}%
\@iflatexlater{2017/01/01}{}{%
    \ClassError{ucasthesis}{TeXLive/MikTex >= 2017 is required ...}{}%
}
\ProvidesClass{ucasthesis}[2014/10/01 v1.0 LaTeX document class]%
%---------------------------------------------------------------------------%
%->> Declare options
%---------------------------------------------------------------------------%
%-
%-> Layout
%-
\DeclareOption{oneside}{%
    \PassOptionsToClass{oneside}{ctexbook}%
}
\DeclareOption{twoside}{%
    \PassOptionsToClass{twoside}{ctexbook}%
}
\newif\ifucas@print \ucas@printfalse
\DeclareOption{print}{%
    \PassOptionsToClass{twoside}{ctexbook}%
    \ucas@printtrue%
}
%-
%-> Language
%-
\newif\ifucas@plain \ucas@plainfalse
\DeclareOption{scheme=plain}{% enable plain writing style
    \PassOptionsToClass{scheme=plain}{ctexbook}%
    \ucas@plaintrue%
}
%-
%-> Draft version info
%-
\newif\ifucas@versioninfo \ucas@versioninfofalse
\DeclareOption{draftversion}{%
    \ucas@versioninfotrue%
}
%-
%-> Handle non-implemented options
%-
\DeclareOption*{%
    \PassOptionsToClass{\CurrentOption}{ctexbook}%
}
%-
%-> Terminates all options processing
%-
\ProcessOptions\relax%
%---------------------------------------------------------------------------%
%->> Load class information
%---------------------------------------------------------------------------%
\LoadClass[UTF8,a4paper,zihao=-4]{ctexbook}%
\@ifclasslater{ctexbook}{2017/01/01}{}{%
    \ClassError{ucasthesis}{ctex >= 2017/01/01 is required ...}{}%
}
%---------------------------------------------------------------------------%
%->> Required packages
%---------------------------------------------------------------------------%
\RequirePackage{ifxetex}% LaTeX engine detection
\RequirePackage{etoolbox}% a toolbox of programming facilities
\newcommand{\ucasifstreq}{\expandafter\ifstrequal\expandafter}% expansion control
\newcommand{\ucasifstrbk}{\expandafter\ifblank\expandafter}% expansion control
\RequirePackage{expl3}% LaTeX3 programming environment
\ExplSyntaxOn
\newcommand{\ucaslntotab}[2][1pt]{% convert lines into table lines
    \seq_set_split:NnV \l_tmpa_seq {\\} {#2}
    \clist_set_from_seq:NN \l_tmpa_clist \l_tmpa_seq
    \clist_clear:N \l_tmpb_clist
    \clist_map_inline:Nn \l_tmpa_clist
      {%
        \clist_put_right:Nn \l_tmpb_clist
          { \ulenhance[#1]{\ulextend{##1}} }
      }
    %\clist_show:N\l_tmpb_clist% show the content of comma list
    \clist_use:Nn \l_tmpb_clist { \\ & }
}
\ExplSyntaxOff
%---------------------------------------------------------------------------%
%->> Load class configuration
%---------------------------------------------------------------------------%
\AtEndOfPackage{% class cfg loaded after package to make preamble commands take effect
    \makeatletter
    \InputIfFileExists{Style/ucasthesis.cfg}{}{}
    \makeatother
}
%---------------------------------------------------------------------------%
%->> Page layout
%---------------------------------------------------------------------------%
%- use \topmargin insead of \voffset to shift the complete text vertically
%- use \<odd|even>sidemargin insead of \hoffset to shift text horizontally
%- part one -- horizontal widths
%- left side width + textwidth + right side width = paperwidth
%- binding side width + textwidth + nonbinding side width = paperwidth
%- binding side width of [odd, even] page = [left, right] side width
%- left side width of [odd, even] page = 1.0in (fixed) + hoffset + [odd, even]sidemargin
%- assuming A4 paper (210mm x 297mm)
\setlength{\textwidth}{146.6mm}% set required text width first
\setlength{\hoffset}{0mm}% set horizontal offset
\ifucas@print% if print layout enabled
    \setlength{\oddsidemargin}{12.6mm}% binding side margin
    \setlength{\evensidemargin}{0mm}% ensure uniform binding side width for printing
\else
    \setlength{\oddsidemargin}{6.3mm}% left side margin
    \setlength{\evensidemargin}{6.3mm}% ensure uniform left side width for EThesis
\fi
\setlength{\marginparwidth}{35pt}% width of margin notes
\setlength{\marginparsep}{10pt}% width of space between body text and margin notes
%- part two -- vertical heights
%- top height + textheight + bottom height = paperheight
%- top height = 1.0in (fixed) + voffset + topmargin + headheight + headsep
\setlength{\textheight}{246.2mm}% set required text height first
\setlength{\voffset}{0mm}% set vertical offset
\setlength{\topmargin}{-10.0mm}% vertical margin above header. 3.0版本调整
\setlength{\headheight}{24pt}% header height. 3.0版本调整
\setlength{\headsep}{5.5pt}% vertical margin between header and body text. 3.0版本调整
\setlength{\footskip}{10.4mm}% vertical margin between footer and body text
%- specifies the amount of space between paragraphs.
\setlength{\parskip}{0.5ex plus 0.25ex minus 0.25ex}
%- line spacing
\linespread{1.5}% line space setting
\raggedbottom% prevent adding vertical white space in strange places
%- default pagestyle is page number at bottom without headers and footers
\pagestyle{plain}
%---------------------------------------------------------------------------%
%->> Style control commands
%---------------------------------------------------------------------------%
\def\NENU@underline[#1]#2{
  \underline{\hbox to #1{\hfill#2\hfill}}}
\def\NENUunderline{\@ifnextchar[\NENU@underline\underline}
\newcommand\Nspace{\protect\hspace{1em}}
\newcommand\NENUspace{\protect\hspace{1em}\protect\hspace{1em}}
%- redefine cleardoublepage to have page style argument
\renewcommand{\cleardoublepage}[1][plain]{%
    \clearpage\if@twoside\ifodd\c@page\else%
    \thispagestyle{#1}%
    \hbox{}\newpage\if@twocolumn\hbox{}\newpage\fi\fi\fi%
}
%- underline
\ifxetex%
    \RequirePackage{ulem}%
\else%
    \RequirePackage{ulem}%
\fi
\newcommand{\ulunify}[1]{\uline{#1}}% unified name
\newcommand{\ulenhance}[2][1pt]{% enhanced underline
    \def\ULthickness{#1}% set thickness
    \setlength{\ULdepth}{0.5em}% set depth
    \expandafter\ulunify\expandafter{#2}% linebreak fix
}
\newcommand{\ulhshift}{-4em}% horizontal shift on underline
\newcommand{\ulextend}[2][132pt]{% extend underline length.3.0修改长度
    \hbox to #1{\hfill\hspace*{\ulhshift}#2\hfill}}
%---------------------------------------------------------------------------%
%->> Titlepage
%---------------------------------------------------------------------------%
%-
%-> Chinese item commands
%-
\def\ucas@value@ch@confidential{}
\newcommand{\confidential}[1]{\def\ucas@value@ch@confidential{#1}}
\def\ucas@value@ch@orderid{}
\newcommand{\orderid}[1]{\def\ucas@value@ch@orderid{#1}}
\def\ucas@value@ch@classid{}
\newcommand{\classid}[1]{\def\ucas@value@ch@classid{#1}}
\def\ucas@value@ch@udc{}
\newcommand{\udc}[1]{\def\ucas@value@ch@udc{#1}}
\def\ucas@value@ch@schoollogo{}
\newcommand{\schoollogo}[2][]{\def\ucas@value@ch@schoollogo{\ucasifstrbk{#1}{\zihao{-2}\bfseries\sffamily #2}{\includegraphics[#1]{#2}}}}
\def\ucas@value@ch@title{}
\def\ucas@value@ch@titlemark{}
\renewcommand{\title}[2][\ucas@value@ch@title]{%
    \def\ucas@value@ch@title{#2}
    %\def\ucas@value@ch@titlemark{\MakeUppercase{#1}}}
    \def\ucas@value@ch@titlemark{#1}}
\def\ucas@value@ch@author{}
\renewcommand{\author}[1]{\def\ucas@value@ch@author{#1}}
\def\ucas@value@ch@advisor{}
\newcommand{\advisor}[1]{\long\def\ucas@value@ch@advisor{#1}}
\def\ucas@value@ch@advisors{}
\newcommand{\advisors}[1]{\long\def\ucas@value@ch@advisors{#1}}
\def\ucas@value@ch@degree{}
\newcommand{\degree}[1]{\def\ucas@value@ch@degree{#1}}
\def\ucas@value@ch@discipline{}
\newcommand{\discipline}[1]{\def\ucas@value@ch@discipline{#1}}
\def\ucas@value@ch@subject{}
\newcommand{\subject}[1]{\def\ucas@value@ch@subject{#1}}
\def\ucas@value@ch@major{}
\newcommand{\major}[1]{\def\ucas@value@ch@major{#1}}
\def\ucas@value@ch@institute{}
\newcommand{\institute}[1]{\long\def\ucas@value@ch@institute{#1}}
\def\ucas@value@ch@school{}
\newcommand{\school}[1]{\def\ucas@value@ch@school{#1}}
\def\ucas@value@ch@orginization{}
\newcommand{\orginization}[1]{\def\ucas@value@ch@orginization{#1}}
\def\ucas@value@ch@theslanguage{}
\newcommand{\theslanguage}[1]{\def\ucas@value@ch@theslanguage{#1}}
\def\ucas@value@ch@thestype{}
\newcommand{\thestype}[1]{\def\ucas@value@ch@thestype{#1}}
\def\ucas@value@ch@completedate{}
\newcommand{\completedate}[1]{\def\ucas@value@ch@completedate{#1}}
\def\ucas@value@ch@submitdate{}
\newcommand{\submitdate}[1]{\def\ucas@value@ch@submitdate{#1}}
\def\ucas@value@ch@studydatefrom{}
\newcommand{\studydatefrom}[1]{\def\ucas@value@ch@studydatefrom{#1}}
\def\ucas@value@ch@studydateto{}
\newcommand{\studydateto}[1]{\def\ucas@value@ch@studydateto{#1}}
\def\ucas@value@ch@date{}
\renewcommand{\date}[1]{\def\ucas@value@ch@date{#1}}
\def\ucas@value@ch@udc{}
\newcommand{\udcnumber}[1]{\def\ucas@value@ch@udc{#1}}
\def\ucas@value@ch@serialnumber{}
\newcommand{\serialnumber}[1]{\def\ucas@value@ch@serialnumber{#1}}
\def\ucas@value@ch@field{}
\newcommand{\field}[1]{\def\ucas@value@ch@field{#1}}
%-
%-> English item commands
%-
\def\ucas@value@en@confidential{}
\newcommand{\CONFIDENTIAL}[1]{\def\ucas@value@en@confidential{#1}}
\def\ucas@value@en@title{}
\def\ucas@value@en@titlemark{}
\newcommand{\TITLE}[2][\ucas@value@en@title]{%
    \def\ucas@value@en@title{#2}
    %\def\ucas@value@en@titlemark{\MakeUppercase{#1}}}
    \def\ucas@value@en@titlemark{#1}}
\def\ucas@value@en@author{}
\newcommand{\AUTHOR}[1]{\def\ucas@value@en@author{#1}}
\def\ucas@value@en@advisor{}
\newcommand{\ADVISOR}[1]{\def\ucas@value@en@advisor{#1}}
\def\ucas@value@en@degree{}
\newcommand{\DEGREE}[1]{\edef\ucas@value@en@degree{\zap@space#1 \@empty}}% expand and remove space
\def\ucas@value@en@degreetype{}
\newcommand{\DEGREETYPE}[1]{\def\ucas@value@en@degreetype{#1}}
\def\ucas@value@en@thesistype{}
\newcommand{\THESISTYPE}[1]{\def\ucas@value@en@thesistype{#1}}
\def\ucas@value@en@major{}
\newcommand{\MAJOR}[1]{\def\ucas@value@en@major{#1}}
\def\ucas@value@en@institute{}
\newcommand{\INSTITUTE}[1]{\def\ucas@value@en@institute{#1}}
\def\ucas@value@en@school{}
\newcommand{\SCHOOL}[1]{\def\ucas@value@en@school{#1}}
\def\ucas@value@en@date{}
\newcommand{\DATE}[1]{\def\ucas@value@en@date{#1}}
\def\ucas@value@en@field{}
\newcommand{\FIELD}[1]{\def\ucas@value@en@field{#1}}
%-
%-> 修改中文字体族
%-

\setCJKmainfont[
Path = ./font/,
BoldFont=simhei.ttf,
ItalicFont=simkai.ttf,
BoldItalicFont=simfang.ttf
]{simsun.ttc}
\setCJKfamilyfont{zhsong}[Path = ./font/]{simsun.ttc}
\newcommand{\song}{\CJKfamily{zhsong}}
\setCJKfamilyfont{zhhei}[Path = ./font/,AutoFakeBold]{simhei.ttf}
\newcommand{\hei}{\CJKfamily{zhhei}}
\setCJKfamilyfont{FZXiHei}[Path = ./font/]{FZXH1K.TTF}
\newcommand{\xhei}{\CJKfamily{FZXiHei}}
\setCJKfamilyfont{zhkai}[Path = ./font/]{simkai.ttf}
\newcommand{\kai}{\CJKfamily{zhkai}}
\setCJKfamilyfont{zhfs}[Path = ./font/]{simfang.ttf}
\newcommand{\fs}{\CJKfamily{zhfs}}
\setCJKfamilyfont{FZXingKai}[Path = ./font/]{FZXKK.TTF}
\newcommand{\xkai}{\CJKfamily{FZXingKai}}
\setCJKfamilyfont{zhls}[Path = ./font/]{simli.ttf}

\setCJKfamilyfont{zhyh}[Path = ./font/]{msyh.ttc} %3.0加入新字体微软雅黑
\newcommand{\yhei}{\CJKfamily{zhyh}}
%-
%-> Detect and adjust items
%-
\AtEndPreamble{%
    \ucasifstreq{\ucas@value@ch@degree}{学士}{%
        \def\ucas@label@ch@thesis{\ucas@label@ch@thesis@bac}%
        \def\ucas@label@ch@thesiss{\ucas@label@ch@thesiss@bac}%3.0添加新title
        \def\ucas@label@ch@author{\ucas@label@ch@author@bac}%
        \def\ucas@label@ch@major{\ucas@label@ch@major@bac}%
        \def\ucas@label@ch@serialnumber{\ucas@label@ch@serialnumber@bac}
        \def\ucas@label@ch@institute{\ucas@label@ch@institute@bac}%
        \def\ucas@value@ch@thesistype{\ucas@value@ch@thesistype@bac}%
        \def\ucas@label@en@thesis{\ucas@label@en@thesis@bac}%
        \def\ucas@label@en@thesiss{\ucas@label@en@thesiss@bac}%3.0添加新title
        \def\ucas@label@en@author{\ucas@label@en@author@bac}%
        \def\ucas@label@en@major{\ucas@label@en@major@bac}%
        \def\ucas@label@en@serialnumber{\ucas@label@en@serialnumber@bac}
        \def\ucas@label@en@institute{\ucas@label@en@institute@bac}%
        \def\ucas@value@en@thesistype{\ucas@value@en@thesistype@bac}%
        \def\maketitle{\maketitle@pdc}
        \def\MAKETITLE{\MAKETITLE@pdc}
        \def\makedeclaration{\makedeclaration@pdc}
        \def\nenulogo{nenu_logo}%3.0添加不同logo
    }{%
    \ucasifstreq{\ucas@value@ch@degree}{硕士}{%
    \def\ucas@label@ch@thesiss{\ucas@label@ch@thesiss@mas}%3.0添加新title
        \def\ucas@label@ch@thesis{\ucas@label@ch@thesis@mas}%
        \def\ucas@label@ch@author{\ucas@label@ch@author@mas}%
        \def\ucas@label@ch@major{\ucas@label@ch@major@mas}%
        \def\ucas@label@ch@institute{\ucas@label@ch@institute@mas}%
        \def\ucas@label@ch@institute{\ucas@label@ch@institute@mas}%
        \def\ucas@value@en@thesistype{\ucas@value@en@thesistype@mas}%
        \def\ucas@label@ch@serialnumber{\ucas@label@ch@serialnumber@mas}
        \def\ucas@label@en@thesiss{\ucas@label@en@thesiss@bac}%3.0添加新title%3.2将“@thesiss@mas”修改为“@thesiss@bac”,英文封面“A thesis”调整为“A Thesis”
        \def\ucas@label@en@thesis{\ucas@label@en@thesis@mas}%
        \def\ucas@label@en@author{\ucas@label@en@author@mas}%
        \def\ucas@label@en@major{\ucas@label@en@major@mas}%
        \def\ucas@label@en@institute{\ucas@label@en@institute@mas}%
        \def\ucas@label@en@institute{\ucas@label@en@institute@mas}%
        \def\ucas@value@en@thesistype{\ucas@value@en@thesistype@mas}%
        \def\ucas@label@en@serialnumber{\ucas@label@en@serialnumber@mas}
        \def\maketitle{\maketitle@xpdc}
        \def\MAKETITLE{\MAKETITLE@xpdc}
        \def\makedeclaration{\makedeclaration@xpdc}
        \def\nenulogo{nenu_logo2}%3.0添加不同logo
    }{%
    \ucasifstreq{\ucas@value@ch@degree}{博士}{%
        \def\ucas@label@ch@thesiss{\ucas@label@ch@thesiss@doc}%3.0添加新title
        \def\ucas@label@ch@thesis{\ucas@label@ch@thesis@doc}%
        \def\ucas@label@ch@author{\ucas@label@ch@author@doc}%
        \def\ucas@label@ch@major{\ucas@label@ch@major@doc}%
        \def\ucas@label@ch@institute{\ucas@label@ch@institute@doc}%
        \def\ucas@label@ch@institute{\ucas@label@ch@institute@doc}%
        \def\ucas@value@en@thesistype{\ucas@value@en@thesistype@doc}%
        \def\ucas@label@ch@serialnumber{\ucas@label@ch@serialnumber@doc}
        \def\ucas@label@en@thesiss{\ucas@label@en@thesiss@doc}%3.0添加新title
        \def\ucas@label@en@thesis{\ucas@label@en@thesis@doc}%
        \def\ucas@label@en@author{\ucas@label@en@author@doc}%
        \def\ucas@label@en@major{\ucas@label@en@major@doc}%
        \def\ucas@label@en@institute{\ucas@label@en@institute@doc}%
        \def\ucas@label@en@institute{\ucas@label@en@institute@doc}%
        \def\ucas@value@en@thesistype{\ucas@value@en@thesistype@doc}%
        \def\ucas@label@en@serialnumber{\ucas@label@en@serialnumber@doc}
        \def\maketitle{\maketitle@xpdc}
        \def\MAKETITLE{\MAKETITLE@xpdc}
        \def\makedeclaration{\makedeclaration@xpdc}
        \def\nenulogo{nenu_logo1}%3.0添加不同logo
    }{%
    \ucasifstreq{\ucas@value@ch@degree}{博士后}{%
        \def\ucas@label@ch@thesis{\ucas@label@ch@thesis@pdc}%
        \def\ucas@label@ch@author{\ucas@label@ch@author@pdc}%
        \def\ucas@label@ch@major{\ucas@label@ch@major@pdc}%
        \def\ucas@label@ch@institute{\ucas@label@ch@institute@pdc}%
        \def\ucas@value@en@thesistype{\ucas@value@en@thesistype@pdc}%
        \def\ucas@label@en@thesis{\ucas@label@en@thesis@pdc}%
        \def\ucas@label@en@author{\ucas@label@en@author@pdc}%
        \def\ucas@label@en@major{\ucas@label@en@major@pdc}%
        \def\ucas@label@en@institute{\ucas@label@en@institute@pdc}%
        \def\ucas@value@en@thesistype{\ucas@value@en@thesistype@pdc}%
        \def\maketitle{\maketitle@pdc}
        \def\MAKETITLE{\MAKETITLE@pdc}
        \def\makedeclaration{\makedeclaration@pdc}
    }{%
        \ClassWarning{ucasthesis}{Invalid value of '\string\degree', please check the spelling}%
    }}}}%
}
%-
%-> Define Chinese style
%-
\newcommand{\maketitle@xpdc}{%
    \cleardoublepage[empty]
    \thispagestyle{empty}
    \begin{center}
    \vspace*{20pt}
        {\zihao{4}\ucas@label@ch@thesiss}\\
        \vspace*{6pt}
{\zihao{5}\ucas@label@ch@udc\ucas@value@ch@udc\qquad\ucas@label@ch@serialnumber\ucas@value@ch@serialnumber
\qquad\ucas@label@ch@confidential\ucas@value@ch@confidential}\\
\vspace*{-6pt}
{\includegraphics[width=14.76cm]{line}}\\
\vspace*{70pt}
        {\ucas@value@ch@schoollogo}\\
\vspace*{28pt}
        {\yhei\zihao{3}\sffamily{\ucas@value@ch@title}}
\vspace*{160pt}

        \def\tabcolsep{3pt}
        \def\arraystretch{0.8}
        \begin{tabular}{lc}
            {\songti\zihao{-4}\ucas@label@ch@author} & \ulenhance[0.5pt]{\ulextend{{\qquad\qquad\songti\zihao{-4}\ucas@value@ch@author}}}\\
           {\songti\zihao{-4} \ucas@label@ch@advisor} & \songti\zihao{-4}{\ucaslntotab[0.5pt]{\ucas@value@ch@advisor}}\\
           % \ucaslntotab[0.5pt]{\ulextend{\qquad\qquad\songti\zihao{-4}{\ucas@value@ch@advisor}}}\\
           {\songti\zihao{-4} \ucas@label@ch@discipline} & \ulenhance[0.5pt]{\ulextend{{\qquad\qquad\songti\zihao{-4}\ucas@value@ch@discipline}}}\\
           {\songti\zihao{-4} \ucas@label@ch@major} & \ulenhance[0.5pt]{\ulextend{{\qquad\qquad\songti\zihao{-4}\ucas@value@ch@major}}}\\
           {\songti\zihao{-4} \ucas@label@ch@field} & \ucaslntotab[0.5pt]{{\qquad\qquad\songti\zihao{-4}\ucas@value@ch@field}}\\
        \end{tabular} \\
\vspace*{45pt}

$\vcenter{\hbox{\includegraphics[scale=0.55]{title1}}}$ \ {\songti\zihao{-4}学位评定委员会}\\
\vspace*{12pt}
        {\songti\zihao{4}\ucas@value@ch@date}

    \end{center}
    \clearpage
 %   \if@twoside
%        \thispagestyle{empty}
%        \ifucas@versioninfo
%            \vspace*{\stretch{1}}
%            \begin{footnotesize}
%                \noindent
%                Draft Version (\today)
%            \end{footnotesize}
%        \fi
%        \cleardoublepage[empty]
%    \else
%        \ifucas@versioninfo
%            \thispagestyle{empty}
%            \vspace*{\stretch{1}}
%            \begin{footnotesize}
%                \noindent
%                Draft Version (\today)
%            \end{footnotesize}
%            \cleardoublepage[empty]
%        \fi
%    \fi
%    \ifucas@print%
%        \hypersetup{% set hyperlinks
%            hidelinks,% remove link color and border
%        }
%    \fi
}

\newcommand{\maketitle@pdc}{%
    \cleardoublepage[empty]
    \thispagestyle{empty}
    \begin{center}
    \vspace*{45pt}
        {\zihao{4}\ucas@label@ch@thesis}\\        
\vspace*{-1pt}
{\zihao{5}\ucas@label@ch@udc\ucas@value@ch@udc\qquad\ucas@label@ch@serialnumber\ucas@value@ch@serialnumber}\\
\vspace*{-12pt}
{\includegraphics[width=14.76cm]{line}}\\
\vspace*{70pt}
        {\ucas@value@ch@schoollogo}\\
\vspace*{22pt}
        {\yhei\zihao{3}\sffamily{\ucas@value@ch@title}}\\[12pt]
        {\yhei\zihao{3}\sffamily{\ucas@value@en@title}}
\vspace*{58pt}

        \def\tabcolsep{3pt}
        \def\arraystretch{0.8}
        \begin{tabular}{lc}
            {\songti\zihao{-4}\ucas@label@ch@author} & \ulenhance[0.5pt]{\ulextend{{\qquad\qquad\songti\zihao{-4}\ucas@value@ch@author}}}\\
           {\songti\zihao{-4} \ucas@label@ch@advisor} & \ucaslntotab[0.5pt]{\ulextend{\qquad\qquad\songti\zihao{-4}\ucas@value@ch@advisor}}\\
           {\songti\zihao{-4} \ucas@label@ch@institute} & \ulenhance[0.5pt]{\ulextend{{\qquad\qquad\songti\zihao{-4}\ucas@value@ch@institute}}}\\
           {\songti\zihao{-4} \ucas@label@ch@major} & \ulenhance[0.5pt]{\ulextend{{\qquad\qquad\songti\zihao{-4}\ucas@value@ch@major}}}\\
           {\songti\zihao{-4} \ucas@label@ch@field} & \ucaslntotab[0.5pt]{{\qquad\qquad\songti\zihao{-4}\ucas@value@ch@field}}\\
           {\songti\zihao{-4} \ucas@label@ch@language} & \ulenhance[0.5pt]{\ulextend{{\qquad\qquad\songti\zihao{-4}\ucas@value@ch@theslanguage}}}\\
           {\songti\zihao{-4} \ucas@label@ch@thestype} & \ulenhance[0.5pt]{\ulextend{{\qquad\qquad\songti\zihao{-4}\ucas@value@ch@thestype}}}\\
        \end{tabular} \\
\vspace*{70pt}

$\vcenter{\hbox{\includegraphics[scale=0.55]{title1}}}$ \ {\songti\zihao{5}学位评定委员会}\\
\vspace*{8pt}
        {\songti\zihao{4}\ucas@value@ch@date}

    \end{center}
        \clearpage
%         %240522修改本科论文封面注释备份↑
%     \cleardoublepage
%   \thispagestyle{empty}
% \hspace{2.5em}{\renewcommand{\baselinestretch}{1}\zihao{-4}\bfseries\ucas@label@ch@udc\underline{\bfseries\ucas@value@ch@udc}}
% \hspace{7.5em}{\zihao{-4}\bfseries\ucas@label@ch@serialnumber\underline{\bfseries\textbf\ucas@value@ch@serialnumber}}

%   \vskip \stretch{2}
% \begin{figure*}[h]
% 			\centering
% 			\includegraphics[width=3.15cm]{nenu_logo.pdf}
% 		\end{figure*}
%   \vskip \stretch{-1}
% \begin{figure*}[h]
% 			\centering
% 			\includegraphics[width=4.45cm]{title.pdf}
% 		\end{figure*}
%   \begin{center}
%   \vskip \stretch{-0.5}
%    {\heiti\zihao{-1} \ucas@label@ch@thesis}
%    \end{center}
%  \vskip \stretch{2}
% \begin{center}
% {\songti\zihao{2} \bfseries\ucas@value@ch@title}
% \end{center}
% \vskip \stretch{0.5}
% \begin{center}
% {\renewcommand{\baselinestretch}{2}\zihao{3} \bfseries\ucas@value@en@title}
% \end{center}
%  \vskip \stretch{2.5}

% \hspace{9em}{\songti\zihao{-3}\ucas@label@ch@author\hspace{1em}\ucas@value@ch@author}


% \hspace{9em}{\songti\zihao{-3}\ucas@label@ch@advisor\hspace{1em}\ucas@value@ch@advisor}

% \hspace{9em}{\songti\zihao{-3}\ucas@label@ch@institute\hspace{1em}\ucas@value@ch@institute}

% \hspace{9em}{\songti\zihao{-3}\ucas@label@ch@major\hspace{1em}\ucas@value@ch@major}

% \ \\
% \ \\

% \begin{center}
%   {\renewcommand{\baselinestretch}{1}\heiti\zihao{4}
%      东北师范大学
%  \vskip \stretch{0}
%     \ucas@value@ch@date}
%   \end{center}
%     \clearpage
%     \if@twoside
%     \thispagestyle{empty}
% %    \ifNENU@typeinfo
%       \vspace*{\stretch{1}}
%       \begin{footnotesize}
%         \noindent
%       \end{footnotesize}
% %    \fi
%     \cleardoublepage
%    \fi
}
%-
%-> Define English style
%-

\newcommand{\MAKETITLE@xpdc}{%
    \cleardoublepage[empty]
    \thispagestyle{empty}
    \begin{center}
    \vspace*{20pt}
       {\zihao{4}\ucas@label@en@thesiss}\\
        \vspace*{6pt}
{\zihao{5}\ucas@label@en@udc\ucas@value@ch@udc\qquad\ucas@label@en@serialnumber\ucas@value@ch@serialnumber
\qquad\ucas@label@en@confidential\ucas@value@en@confidential}\\
\vspace*{-6pt}
{\includegraphics[width=14.76cm]{line}}\\
\vspace*{70pt}
        {\ucas@value@ch@schoollogo}\\
\vspace*{28pt}
        {\zihao{3}\textbf{\ucas@value@en@title}}
\vspace*{160pt}

        \def\tabcolsep{3pt}
        \def\arraystretch{0.8}
        \begin{tabular}{rc}
            {\songti\zihao{-4}\ucas@label@en@author} & \ulenhance[0.5pt]{\ulextend{{\qquad\qquad\songti\zihao{-4}\ucas@value@en@author}}}\\
           {\songti\zihao{-4} \ucas@label@en@advisor} & \songti\zihao{-4}{\ucaslntotab[0.5pt]{\ucas@value@en@advisor}}\\
           % \ucaslntotab[0.5pt]{{\qquad\qquad\songti\zihao{-4}\ucas@value@en@advisor}}\\
           {\songti\zihao{-4} \ucas@label@en@discipline} & \ulenhance[0.5pt]{\ulextend{{\qquad\qquad\songti\zihao{-4}\ucas@value@en@degreetype}}}\\
           {\songti\zihao{-4} \ucas@label@en@major} & \ulenhance[0.5pt]{\ulextend{{\qquad\qquad\songti\zihao{-4}\ucas@value@en@major}}}\\
           {\songti\zihao{-4} \ucas@label@en@field} & \ucaslntotab[0.5pt]{{\qquad\qquad\songti\zihao{-4}\ucas@value@en@field}}\\
        \end{tabular} \\
\vspace*{45pt}

$\vcenter{{\songti\zihao{-4}Northeast Normal University Academic Degree Evaluation Committee}}$\\
\vspace*{12pt}
        {\songti\zihao{4}\ucas@value@en@date}

    \end{center}
    \clearpage
 %   \if@twoside
%        \thispagestyle{empty}
%        \ifucas@versioninfo
%            \vspace*{\stretch{1}}
%            \begin{footnotesize}
%                \noindent
%                Draft Version (\today)
%            \end{footnotesize}
%        \fi
%        \cleardoublepage[empty]
%    \else
%        \ifucas@versioninfo
%            \thispagestyle{empty}
%            \vspace*{\stretch{1}}
%            \begin{footnotesize}
%                \noindent
%                Draft Version (\today)
%            \end{footnotesize}
%            \cleardoublepage[empty]
%        \fi
%    \fi
%    \ifucas@print%
%        \hypersetup{% set hyperlinks
%            hidelinks,% remove link color and border
%        }
%    \fi
}
\newcommand{\MAKETITLE@pdc}{}

%---------------------------------------------------------------------------%
%->> Author's declaration
%---------------------------------------------------------------------------%
\newcommand{\makedeclaration@xpdc}{%     %论文的独创性说明
    \cleardoublepage[empty]
    \thispagestyle{empty}
    \begin{table}[h]
\centering
\begin{tabular}{|p{410pt}|}
\hline
    \begin{center}
        \vspace*{0cm}
            {\zihao{-3}\bfseries \ucas@label@ch@declare@creativity}
    \end{center}

    {\renewcommand{\baselinestretch}{1.5}\zihao{-4}~~~~~~~~\ucas@value@ch@declare@creativity}
\ \\

    {\zihao{-4}~~~~~~~~学位\ucas@label@ch@declare@author}\NENUunderline[8em]{}
    ~~{\zihao{-4}\ucas@label@ch@declare@date}\NENUunderline[7em]{} \\
\ \\
\ \\
\ \\
\ \\
\ \\
\ \\
%%%%%%%%%%%%%%%%%学位论文使用授权书%%%%%%%%%%%%%
\begin{center}
            \vspace*{2cm}
            {\zihao{-3} \bfseries\ucas@label@ch@declare@rights}
    \end{center}

    {\renewcommand{\baselinestretch}{1.5}\zihao{-4}~~~~~~~\ucas@value@ch@declare@rights}
    \vspace*{0.1cm}
\ \\

    {\zihao{-4}~~~~~~~~\ucas@label@ch@declare@author}\NENUunderline[4em]{}\hspace{3.5em}
    {\zihao{-4}\ucas@label@ch@declare@advisor}\NENUunderline[6em]{}

        {\zihao{-4}~~~~~~~~\ucas@label@ch@authorsigndate}\NENUunderline[4em]{}
    \hspace{3.5em}{\zihao{-4}\ucas@label@ch@advisorsigndate}\NENUunderline[6em]{}
\ \\
\  \\
 %%%%%%%%%%%%%%%%%%%%%%%%毕业去向%%%%%%%%%%%%%%%%%%%%
 % {\renewcommand{\baselinestretch}{1.5}{~~~~~~~~\zihao{-4}\ucas@label@ch@job}}

 %       {~~~~~~~~\zihao{-4}\ucas@label@ch@jobaffliation}\NENUunderline[14em]{}
 %    \hspace{3em}{\zihao{-4}\ucas@label@ch@jobphone}\NENUunderline[4.5em]{}

 %    {~~~~~~~~\zihao{-4}\ucas@label@ch@jobaddress}\NENUunderline[14em]{}
 %    \hspace{3em}{\zihao{-4}\ucas@label@ch@jobpostcode}\NENUunderline[4.5em]{} \\
    \ \\%240522去掉全部非博后毕业去向，加入两个空行"\ \\"
    \ \\
    \ \\
\hline
\end{tabular}
\end{table}
    \clearpage
    \if@twoside
        \thispagestyle{empty}
        \cleardoublepage[empty]
    \fi
}

\newcommand{\makedeclaration@pdc}{%     %论文的独创性说明 学士
    \cleardoublepage[empty]
    \thispagestyle{empty}
\begin{center}
        \vspace*{0cm}
            {\zihao{-3}\bfseries \ucas@label@ch@declare@creativity}
    \end{center}

    {\renewcommand{\baselinestretch}{1.5}\zihao{-4}\ucas@value@ch@declare@creativity}
\ \\

    {\zihao{-4}\ucas@label@ch@declare@author}\NENUunderline[8em]{}
    ~~{\zihao{-4}\ucas@label@ch@declare@date}\NENUunderline[7em]{} \\
    \clearpage
    \if@twoside
        \thispagestyle{empty}
        \cleardoublepage[empty]
    \fi
}

%---------------------------------------------------------------------------%
%->> New environments
%---------------------------------------------------------------------------%
%- define chinese keywords
\newcommand{\keywords}[1]{%3.0修改关键词首行缩进
    \vspace{\baselineskip}
    %\noindent
    {\bfseries \ucas@label@ch@keywords}
     #1}
%- define engish keywords
\newcommand{\KEYWORDS}[1]{%
    \vspace{\baselineskip}
    \noindent {\bfseries \ucas@label@en@keywords} #1}
%---------------------------------------------------------------------------%
%->> Structure elements
%---------------------------------------------------------------------------%
\ifucas@plain%
    \def\contentsname{\ucas@label@en@tocname}
    \def\listfigurename{\ucas@label@en@lsfigname}
    \def\listtablename{\ucas@label@en@lstabname}
    \def\bibname{\ucas@label@en@bibname}
    \def\refname{\ucas@label@en@bibname}
    \def\algname{\ucas@label@en@algname}
    \def\bibetal{\ucas@label@en@bibetal}
    \def\biband{\ucas@label@en@biband}
    \def\axiomname{\ucas@label@en@axiomname}
    \def\theoremname{\ucas@label@en@theoremname}
    \def\lemmaname{\ucas@label@en@lemmaname}
    \def\corollaryname{\ucas@label@en@corollaryname}
    \def\assertionname{\ucas@label@en@assertionname}
    \def\propositionname{\ucas@label@en@propositionname}
    \def\conjecturename{\ucas@label@en@conjecturename}
    \def\definitionname{\ucas@label@en@definitionname}
    \def\examplename{\ucas@label@en@examplename}
    \def\remarkname{\ucas@label@en@remarkname}
    \def\proofname{\ucas@label@en@proofname}
    \def\@title{\ucas@value@en@titlemark}
    \def\@author{\ucas@value@en@author}
\else
    \def\contentsname{\ucas@label@ch@tocname}
    \def\listfigurename{\ucas@label@ch@lsfigname}
    \def\listtablename{\ucas@label@ch@lstabname}
    \def\algname{\ucas@label@ch@algname}
    \def\bibname{\ucas@label@ch@bibname}
    \def\refname{\ucas@label@ch@bibname}
    \def\bibetal{\ucas@label@ch@bibetal}
    \def\biband{\ucas@label@ch@biband}
    \def\axiomname{\ucas@label@ch@axiomname}
    \def\theoremname{\ucas@label@ch@theoremname}
    \def\lemmaname{\ucas@label@ch@lemmaname}
    \def\corollaryname{\ucas@label@ch@corollaryname}
    \def\assertionname{\ucas@label@ch@assertionname}
    \def\propositionname{\ucas@label@ch@propositionname}
    \def\conjecturename{\ucas@label@ch@conjecturename}
    \def\definitionname{\ucas@label@ch@definitionname}
    \def\examplename{\ucas@label@ch@examplename}
    \def\remarkname{\ucas@label@ch@remarkname}
    \def\proofname{\ucas@label@ch@proofname}
    \def\@title{\ucas@value@ch@titlemark}
    \def\@author{\ucas@value@ch@author}
\fi
%---------------------------------------------------------------------------%
%->> Structure layout   %3.0重新修改字体
%---------------------------------------------------------------------------%
\ctexset {
    chapter = {
        format = \linespread{1.5}\zihao{3}\heiti\sffamily\centering,
        name = {}, %3.0修改章标题样式
        %nameformat = {},
        titleformat = {},
        number = \arabic{chapter},
        numberformat = \rmfamily,
        aftername = \quad,
        beforeskip = {20pt},
        afterskip = {24pt},
        pagestyle = plain,
    }
}
%- section
\ctexset {
    section = {
        format = \linespread{1.5}\zihao{4}\sffamily\raggedright,
        numberformat = \rmfamily,
        aftername = \quad,
        beforeskip = {6pt},
        afterskip = {0pt},
    }
}
%- subsection
\ctexset {
    subsection = {
        format = \linespread{1.5}\zihao{-4}\sffamily\raggedright,
        numberformat = \rmfamily,
        aftername = \quad,
        beforeskip = {6pt},
        afterskip = {0pt},
    }
}
%- subsubsection
\ctexset {
    subsubsection = {
        format = \linespread{1.0}\zihao{-4}\sffamily\raggedright,
        numberformat = \rmfamily,
        aftername = \quad,
        beforeskip = {12pt},
        afterskip = {6pt},
    }
}
%- appendix
\ctexset {
    appendix = {
        %numbering = true|false,
        %number = \rmfamily\Alph{chapter},
    }
}
%---------------------------------------------------------------------------%
%->> Configure table of contents
%---------------------------------------------------------------------------%
%- the table of contents is specified by defining \l@chapter, \l@section, ...
%- most commands are then defined with the \@dottedtocline:
%- \@dottedtocline{⟨level⟩}{⟨indent⟩}{⟨numwidth⟩}{⟨title⟩}{⟨page⟩}
%- ⟨level⟩    an entry is produced only if ⟨level⟩ <= value of the tocdepth counter
%-            note, \chapter is level 0, \section is level 1, etc
%- ⟨indent⟩   indentation from the outer left margin of the start of the contents line
%- ⟨numwidth⟩ width of a box in which the section number is to go
%- <title>    title
%- <page>     page number

%- parameters for spacing and length in toc
\def\@dotsep{1.5mu}% separation between dots in mu units
\def\@pnumwidth{2em}% width of a box in which the page number is put
\def\@tocrmarg{2em}% right margin for multiple line entries, \@tocrmarg ≥ \@pnumwidth
\def\@chaptervspace{1ex}% vertical space between chapter titles

%- redefine dottedtocline from classes.dtx and latex.ltx
\renewcommand*{\@dottedtocline}[5]{% [<level>,<indent>,<numwidth>,<title>,<page>]
    \ifnum #1>\c@tocdepth \else
        \vskip \z@ \@plus.2\p@%此行可微调目录行距以确保分页
        \vskip 1.5\@chaptervspace
        {\leftskip #2\relax \rightskip \@tocrmarg \parfillskip -\rightskip
        \parindent #2\relax\@afterindenttrue
        \interlinepenalty\@M
        \leavevmode \zihao{-4}\songti%此行可设置目录中小标题字体字号，240522将“\leavevmode \zihao{-4}\sffamily”修改为“\leavevmode \zihao{-4}\songti”，小标题目录由黑体调整为宋体
        \@tempdima #3\relax
        \advance\leftskip \@tempdima \null\nobreak\hskip -\leftskip
        {#4}\nobreak
        \leaders\hbox{$\m@th\mkern \@dotsep \cdot\mkern \@dotsep$}\hfill
        \nobreak
        \hb@xt@\@pnumwidth{\hfil \zihao{-4} \rmfamily \normalcolor  #5}%
        \par\penalty\@highpenalty}%
    \fi
}
%- redefine l@part from book.cls to add dotted toc line
\renewcommand*{\l@part}[2]{% [<title>,<page>]
    \ifnum \c@tocdepth >-2\relax
        \addpenalty{-\@highpenalty}%
        \addvspace{2.25em \@plus\p@}%%此行调整目录段前行距
        \vskip 1.5\@chaptervspace
        \setlength\@tempdima{3em}%
        \begingroup
            \parindent \z@ \rightskip \@pnumwidth
            \parfillskip -\@pnumwidth
            {\leavevmode
            \zihao{4}\sffamily #1
            \leaders\hbox{$\m@th\mkern \@dotsep \cdot\mkern \@dotsep$}% add dotted toc line
            \hfil \hb@xt@\@pnumwidth{\hss #2}}\par
            \nobreak
            \global\@nobreaktrue
            \everypar{\global\@nobreakfalse\everypar{}}%
        \endgroup
    \fi
}
%- redefine l@chapter from book.cls to add dotted toc line
\renewcommand*{\l@chapter}[2]{% [<title>,<page>]
    \ifnum \c@tocdepth >\m@ne
        \addpenalty{-\@highpenalty}%
        % \vskip \@chaptervspace \@plus\p@%此行可弹性调整目录中chapter的段前行距以照顾分页，240522为确保目录段前段后0行注释掉
        \vskip 1.5\@chaptervspace
        \setlength\@tempdima{1.5em}%%此行可调节目录chapter文字前缩进
        \begingroup
            \parindent \z@ \rightskip \@pnumwidth
            \parfillskip -\@pnumwidth
            \leavevmode \zihao{-4}\sffamily
            \advance\leftskip\@tempdima
            \hskip -\leftskip
            #1\nobreak
            \leaders\hbox{$\m@th\mkern \@dotsep \cdot\mkern \@dotsep$}% add dotted toc line
            \hfil \nobreak\hb@xt@\@pnumwidth{\hss #2}\par
            \penalty\@highpenalty
        \endgroup
    \fi
}
%- redefine indentation for others
\renewcommand*\l@section{\@dottedtocline{1}{1em}{1.8em}}
\renewcommand*\l@subsection{\@dottedtocline{2}{2em}{2.8em}}
\renewcommand*\l@subsubsection{\@dottedtocline{3}{3em}{3.8em}}
\renewcommand*\l@paragraph{\@dottedtocline{4}{4em}{4.8em}}
\renewcommand*\l@subparagraph{\@dottedtocline{5}{5em}{5.8em}}
\renewcommand*\l@figure{\@dottedtocline{1}{1em}{1.8em}}
\renewcommand*\l@table{\@dottedtocline{1}{1em}{1.8em}}
%---------------------------------------------------------------------------%
\endinput
