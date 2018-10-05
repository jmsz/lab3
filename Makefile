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
	curl -L -o ./data/cs_energies.txt "https://www.dropbox.com/s/yxm9lbnailhhdao/cs_energies.txt?dl=0"
	curl -L -o ./data/co_energies.txt "https://www.dropbox.com/s/99cfeid7nto0k35/co_energies.txt?dl=0"
	curl -L -o ./data/am_energies.txt "https://www.dropbox.com/s/glh5hohb924bfey/am_energies.txt?dl=0"
# Run tests on analysis scripts
test :
	python ./scripts/test.py

# Automate running of sample parts of the analysis.
analysis :
	python ./scripts/lab1_analysis_sample.py

# Automate running the full analysis.
fullanalysis :
	python ./scripts/lab1_analysis.py

clean :
	rm -f *.md5 *.aux *.log *.bbl *.lof *.lot *.blg *.out *.toc *.run.xml *.bcf *.txt
	rm -f text/*.aux
	rm *.pdf
	rm $(manuscript).pdf
	rm scripts/*.pyc

# Make keyword for commands that don't have dependencies
.PHONY : test data validate analysis clean
