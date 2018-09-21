manuscript = report
latexopt = -file-line-error -halt-on-error

# Build the PDF of the lab report from the source files
$(manuscript).pdf: $(manuscript).tex text/*.tex references.bib images/*.pdf
	pdflatex $(latexopt) $(manuscript).tex
	bibtex $(manuscript).aux
	bibtex $(manuscript).aux
	pdflatex $(latexopt) $(manuscript).tex
	pdflatex $(latexopt) $(manuscript).tex

# Get/download necessary data
data :
	curl -L -o lab0_spectral_data.txt "https://www.dropbox.com/s/hutmwip3681xlup/lab0_spectral_data.txt?dl=0"

# Validate that downloaded data is not corrupted
validate :
	curl -L -o lab0_spectral_data.md5 "https://www.dropbox.com/s/amumdrm9zp1kn8d/lab0_spectral_data.md5?dl=0"
	md5sum -c lab0_spectral_data.md5

# Run tests on analysis scripts
test :
	python test/test_lab0analysis.py
	#nosetests --no-byte-compile test/*

# Automate running the analysis scripts
analysis :
	python scripts/lab0analysis.py

clean :
	rm -f *.md5 *.aux *.log *.bbl *.lof *.lot *.blg *.out *.toc *.run.xml *.bcf *.txt
	rm -f text/*.aux
	rm ./test/*.pdf
	rm $(manuscript).pdf
	rm scripts/*.pyc

# Make keyword for commands that don't have dependencies
.PHONY : test data validate analysis clean
