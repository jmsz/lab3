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
	curl -L -o ./data/hits.h5 "https://www.dropbox.com/s/ojq4i9fyz8f7205/hits.h5?dl=0"
	curl -L -o ./data/calibration_long.txt "https://www.dropbox.com/s/nkbwd0iie6fp623/calibration_long.txt?dl=0"
	curl -L -o ./data/cs2.h5 "https://www.dropbox.com/s/5fl7muuiqxilg25/cs2.h5?dl=0"

# Run tests on analysis scripts
test :
	python ./scripts/test.py

# Automate running the full analysis.
analysis :
	python ./calculateWindow.py
	python ./scripts/timing.py

clean :
	rm -f *.md5 *.aux *.log *.bbl *.lof *.lot *.blg *.out *.toc *.run.xml *.bcf *.txt
	rm -f text/*.aux
	rm *.pdf
	rm $(manuscript).pdf
	rm scripts/*.pyc

# Make keyword for commands that don't have dependencies
.PHONY : test data validate analysis clean
