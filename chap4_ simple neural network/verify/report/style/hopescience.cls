%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Hope's science report
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{hopescience}[2024/7/10 Hope's science report]


% 加载 ctexart 类作为父类
% 该类提供的宏包包括但不限于 \heiti \zihao 等指令
\LoadClass[UTF8, a4paper]{ctexart}


% 设置页边距
\setlength{\headheight}{43pt}
\RequirePackage[a4paper,top=4.2cm,bottom=2.7cm,left=2.0cm,right=2.0cm]{geometry}



% 设置图表编号和章节相关联
\RequirePackage{chngcntr}
\counterwithin{figure}{section}
\counterwithin{table}{section}
\renewcommand{\thefigure}{\thesection-\arabic{figure}\space}


% 设置参考文献格式
\RequirePackage[
    backend=biber,
    style=gb7714-2015,
    natbib=true,
]{biblatex}
\addbibresource{bib/reference.bib}


% 设置正文页章节标题格式
\ctexset{
    section={ % 一级标题
        format={\heiti\zihao{4}}, % 四号黑体居中
        number={第\arabic{section}部分},
        aftername={\quad}, % 序号与标题之间空 4 格
        beforeskip={.5\baselineskip}, % 段前0.5 行
        afterskip={.5\baselineskip}, % 段后0.5 行
    },
    subsection={ % 二级标题
        format={\heiti\zihao{5}}, % 格式: 五号黑体
        aftername={\space}, %序号与题名之间空 1 格
        beforeskip={.5\baselineskip}, % 段前 0.5 行
        afterskip={.5\baselineskip}, % 段后 0.5 行
    },
    subsubsection={ % 三级标题
        format={\heiti\zihao{5}}, % 格式: 五号黑体
        aftername={\space}, % 序号与题目名称间空 1 格
        beforeskip={.5\baselineskip}, % 段前 0.5 行
        afterskip={0.5\baselineskip}, % 段后 0.5 行
        indent={2em}, % 缩进 2 个汉字符
    },
    paragraph={ % 四级标题
        format={\heiti\zihao{5}}, % 五号黑体
        numbering=true, % 启用编号
        number={\Alph{paragraph}}, % 编号格式: 使用大写字母作为四级标题
        aftername={.\space}, % 序号后带点和一个空格
        beforeskip={0pt}, % 段前 0 行
        afterskip={0pt}, % 段后 0 行
        indent={2em}, % 缩进 2 个汉字符
        aftertitle={\par}, % 标题后添加一个空行
    },
    subparagraph={ % 五级标题
        format={\heiti\zihao{5}}, % 格式: 五号黑体
        numbering=true, % 启用编号
        number={\alph{subparagraph}}, % 编号格式: 使用小写字母作为五级标题
        aftername={.\space}, % 序号后带点和一个空格
        beforeskip={0pt}, % 段前 0 行
        afterskip={0pt}, % 段后 0 行
        indent={2em}, % 缩进 2 个汉字符
        aftertitle={\par}, % 标题后添加一个空行
    }
}

