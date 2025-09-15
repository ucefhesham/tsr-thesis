# VS Thesis Template

[![pipeline status](https://gitlab-vs.informatik.uni-ulm.de/templates/thesis-template/badges/master/pipeline.svg)](https://gitlab-vs.informatik.uni-ulm.de/templates/thesis-template/commits/master) [![xelatex pdf](https://img.shields.io/badge/-XeLaTeX-green.svg?logo=adobe-acrobat-reader&style=flat)](https://gitlab-vs.informatik.uni-ulm.de/templates/thesis-template//-/jobs/artifacts/master/file/out/main.pdf?job=build-xelatex) [![pdftex pdf](https://img.shields.io/badge/-pdfTeX-green.svg?logo=adobe-acrobat-reader&style=flat)](https://gitlab-vs.informatik.uni-ulm.de/templates/thesis-template//-/jobs/artifacts/master/file/out/main.pdf?job=build-pdflatex) [![luatex pdf](https://img.shields.io/badge/-LuaTeX-green.svg?logo=adobe-acrobat-reader&style=flat)](https://gitlab-vs.informatik.uni-ulm.de/templates/thesis-template//-/jobs/artifacts/master/file/out/main.pdf?job=build-luatex)


This template is a general template for theses and project reports based on the memoir class with support for pdfTeX, LuaTeX, and XeTeX.
The recommended way to build the template is to use latexmk with the provided latexmkrc that uses XeTeX.

```bash
 $ latexmk main.tex
```

This will continuously build a pdf of your thesis.
All build artifacts and temporary files will be stored in the `out/` directory and can be cleaned up with the following command (use -C to also delete pdf).

```bash
 $ latexmk -c
```

## Configuration
All necessary configuration can be adjusted inside the `main.tex`.

### Thesis Information
 * Title: `\title{Your Thesis Title}`
 * Type: `\thesistype{Master Thesis | Bachelor Thesis | Project Report}`
 * Title Graphic: `\titlegraphic{titleimage.jpg}` (recommended aspect ratio 31:10, can be omitted to show a gray box instead)
 * Submission Date: `\submissiondate{2019-03-14}`
 * License: `\license{\ccby}` (should be set to a command provided by the `ccicons` package or omitted to not provide any licensing information)

### Author Information
 * Author: `\author{Your Name}`
 * Student Id: `\studentnumber{123456}`
 * VS Number: `\internalid{VS-2019-B18}` (Will be provided by your supervisor)
 * Place of signing the "independence declaration": `\place{Ulm}`

### Institute Information
 * Examiners: `\examiners{Prof.\@~Dr.-Ing.\@~Jane Doe}` (multiple examiners should be separated by `\\`)
 * Supervisor: `\supervisors{M.Sc.\@~Arthur Dent}` (multiple supervisors should be separated by `\\`)
 * Institute: `\institution{...}` (usually the default should be fine)

### Language
The template features English and German translations.
To set the language adjust the `\setdefaultlanguage{english}` line in the `main.tex`.
It is currently necessary to manually adjust the `\institution{...}` command, when using the German translation.

### Class Options
 * `lesscolor`: reduces the amount of different colors used for the title page
 * `twoside`: when the document will be published with printing on both sides of the paper

## Further Information
The template is based on [smart-thesis](https://github.com/astoeckel/smart-thesis) to provide a classicthesis inspired look.
It uses FiraSans (similar to the University CI font Meta Pro) for the titlepage and the Cochineal (similar to Minion Pro) font face for the main text body, which are both licensed under SIL Open Font License.
Both fonts should be included in most LaTeX distributions, so that a manual font installation is not required.
When using XeTeX, `fontspec` is used to load the OpenType font variants provided by your LaTeX distribution.
