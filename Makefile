all: paper.pdf

paper.pdf: paper.tex
	pdflatex paper.tex
	bibtex paper.aux
	pdflatex paper.tex
	pdflatex paper.tex

clean:
	rm -rf *.aux *.log *.out *.bbl *.blg
	rm -rf paper.pdf
