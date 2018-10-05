manuscript = report
latexopt = -file-line-error -halt-on-error

# Build the PDF of the lab report from the source files
$(manuscript).pdf: $(manuscript).tex text/*.tex references.bib figures/*
	pdflatex $(latexopt) $(manuscript).tex
	bibtex $(manuscript).aux
	bibtex $(manuscript).aux
	pdflatex $(latexopt) $(manuscript).tex
	pdflatex $(latexopt) $(manuscript).tex

# Get/download necessary data
data :
	curl -L -o ./data/tenevents.txt "https://www.dropbox.com/s/yk9cov2q74ib9fl/tenevents.txt?dl=0"
	curl -L -o ./data/cs_energies.txt "https://www.dropbox.com/s/wqcxnjptgj6x7om/cs_energies.txt?dl=0"
	curl -L -o ./data/co_energies.txt "https://www.dropbox.com/s/njszde9miwqfby9/co_energies.txt?dl=0"
	curl -L -o ./data/am_energies.txt "https://www.dropbox.com/s/e3sqk1pwqrpvld1/am_energies.txt?dl=0"

# Run tests on analysis scripts
test :
	cd ./scripts
	python lab1_analysis_test.py
	cd ..

# Automate running of sample parts of the analysis.
analysis :
	cd ./scripts
	python scripts/lab1_analysis_sample.py
	cd ..

# Automate running the full analysis.
fullanalysis :
	cd ./scripts
	python scripts/lab1_analysis.py
	cd ..
	
clean :
	rm -f *.md5 *.aux *.log *.bbl *.lof *.lot *.blg *.out *.toc *.run.xml *.bcf *.txt
	rm -f text/*.aux
	rm ./test/*.pdf
	rm *.pdf
	rm $(manuscript).pdf
	rm scripts/*.pyc

# Make keyword for commands that don't have dependencies
.PHONY : test data validate analysis clean
